"""Layer 3 view: AI-led structured behavioral interview.

Flow per competency (v6-style, two-step with Continue buttons):
  1. Show the main question. Candidate records (or types) their answer.
  2. LLM picks a follow-up bucket and writes the follow-up.
  3. Candidate records (or types) their follow-up answer.
  4. LLM scores the competency on a 0-25 scale based on both exchanges.
  5. Save row, advance to next competency.

4 competencies total -> Layer 3 score is 0-100.

90-second answer window:
  The 90s timer starts the moment the AI finishes speaking the question
  (we estimate spoken duration from text length client-side). When the
  timer hits zero:
    - if the candidate is mid-recording, the recorder is auto-stopped
      (the existing recording_cap component handles that),
    - if the candidate hasn't started recording yet, we submit an
      empty answer and advance.
"""

from __future__ import annotations

import time

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from assessment_logic.layer3_logic import (
    COMPETENCY_COUNT,
    generate_followup,
    load_main_questions,
    score_competency,
)
from assessment_logic.llm_client import transcribe_audio
from assessment_logic.recording_cap import render_recording_cap
from assessment_logic.tts import speak
from database import db

from .state import advance_stage

# st.audio_input is built into Streamlit and records at 16kHz mono by default,
# which is exactly what gpt-4o-mini-transcribe expects.
MIC_AVAILABLE = hasattr(st, "audio_input")

# Per-answer window in seconds. The clock starts the moment the AI is
# estimated to have finished speaking the question.
ANSWER_WINDOW_SECONDS = 90

# Rough characters-per-second for the browser TTS at rate=0.95. Used to
# estimate when the spoken question will finish so we can start the 90s
# clock at the right moment. A bit conservative so the candidate isn't
# rushed if the browser speaks slightly slower than estimated.
TTS_CHARS_PER_SECOND = 14.0
TTS_MIN_DURATION = 2.0
TTS_MAX_DURATION = 25.0


def render() -> None:
    candidate_id = st.session_state.candidate_id

    if not st.session_state.get("l3_started", False):
        _intro()
        return

    if not st.session_state.l3_main_questions:
        st.session_state.l3_main_questions = load_main_questions(candidate_id)

    comp_idx = st.session_state.l3_question_idx

    if comp_idx >= COMPETENCY_COUNT:
        _finish_layer()
        return

    comp = st.session_state.l3_main_questions[comp_idx]
    phase = st.session_state.l3_phase  # 'main' or 'followup'

    if phase == "main":
        _render_question(
            comp=comp,
            phase="main",
            question_text=comp["question"],
        )
    else:
        followup = st.session_state.l3_current_followup or {}
        _render_question(
            comp=comp,
            phase="followup",
            question_text=followup.get("question", "Can you tell me more about that?"),
        )


def _intro() -> None:
    st.title("Layer 3 — AI-Led Interview")
    st.markdown(
        f"""
        You'll be asked **{COMPETENCY_COUNT} interview questions**. The AI
        interviewer will read each question out loud, then you'll record a
        voice answer. The AI will then ask one follow-up based on what you
        said, and read that out loud too.

        **How it works:**
        1. Listen as the AI reads the question.
        2. Once the AI finishes speaking, you have **{ANSWER_WINDOW_SECONDS}
           seconds** to record your answer.
        3. Click **Start recording**, answer out loud, then click **Stop**
           (or the timer will stop you).
        4. Review the transcript, then continue to the AI's follow-up.

        If you miss a question you can press **🔊 Replay question** at any
        point. If transcription fails, you can type your answer instead.

        **Tips:**
        - Use concrete, specific examples.
        - It's fine to pause and think before you answer.
        - Don't rush. Clarity beats speed.
        - Make sure your speakers or headphones are on.

        Total time: about 16 minutes.
        """
    )

    if not MIC_AVAILABLE:
        st.warning(
            "The voice recorder component isn't available. You'll be able to "
            "type your answers instead."
        )

    if st.button("Begin Layer 3", type="primary", use_container_width=True):
        st.session_state.l3_started = True
        st.session_state.l3_question_started_at = time.time()
        st.rerun()


def _estimate_tts_seconds(text: str) -> float:
    """Estimate how long the browser TTS will take to speak `text`.

    The browser doesn't give us a clean "speak finished" callback we can
    receive on the Python side, so we estimate from text length. The
    estimate is intentionally a bit generous so the 90s answer window
    starts when the candidate is actually done listening, not while the
    AI is still mid-sentence.
    """
    if not text:
        return TTS_MIN_DURATION
    est = len(text) / TTS_CHARS_PER_SECOND
    return max(TTS_MIN_DURATION, min(TTS_MAX_DURATION, est))


def _render_question(comp: dict, phase: str, question_text: str) -> None:
    comp_idx = st.session_state.l3_question_idx
    exchange_num = (comp_idx * 2) + (1 if phase == "main" else 2)
    total_exchanges = COMPETENCY_COUNT * 2

    # Per-question state keys
    transcript_key = f"l3_transcript_{comp_idx}_{phase}"
    audio_bytes_key = f"l3_audio_{comp_idx}_{phase}"
    transcript_shown_key = f"l3_transcript_shown_{comp_idx}_{phase}"
    spoken_key = f"l3_spoken_{comp_idx}_{phase}"
    deadline_key = f"l3_deadline_{comp_idx}_{phase}"
    autoadvanced_key = f"l3_autoadvanced_{comp_idx}_{phase}"

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Layer 3 — Question {comp_idx + 1} of {COMPETENCY_COUNT}**")
        st.progress(
            (exchange_num - 1) / total_exchanges,
            text=f"Exchange {exchange_num} of {total_exchanges}",
        )
    with col2:
        # Live countdown — only shown once the deadline is set (i.e.
        # after the AI is estimated to have finished speaking).
        deadline = st.session_state.get(deadline_key)
        if deadline is not None and not st.session_state.get(transcript_shown_key):
            remaining = max(0, int(deadline - time.time()))
            mins, secs = divmod(remaining, 60)
            color = "🟢" if remaining > 30 else ("🟡" if remaining > 10 else "🔴")
            st.metric("Answer time", f"{color} {mins:02d}:{secs:02d}")

    st.divider()
    heading = f"Question {comp_idx + 1}"
    if phase == "followup":
        heading += " — follow-up"
    st.markdown(f"### {heading}")
    st.info(question_text)

    # Speak the question once when this (comp_idx, phase) is first shown.
    # On reruns we skip autoplay but the Replay button still works.
    autoplay = not st.session_state.get(spoken_key, False)
    speak(question_text, autoplay=autoplay)
    if autoplay:
        st.session_state[spoken_key] = True
        # Set the answer-window deadline to start AFTER the AI finishes
        # speaking. We estimate that duration from the text length.
        est_tts_secs = _estimate_tts_seconds(question_text)
        st.session_state[deadline_key] = time.time() + est_tts_secs + ANSWER_WINDOW_SECONDS

    # Keep ticking once per second so the countdown updates and the
    # auto-advance check below fires on every tick. We only tick while
    # the candidate hasn't yet shown a transcript (review mode pauses
    # the clock).
    if not st.session_state.get(transcript_shown_key):
        st_autorefresh(interval=1000, key=f"l3_tick_{comp_idx}_{phase}")

    # ---- Time-up auto-advance (only when no transcript yet) ----
    # If the deadline has passed and the candidate has not produced a
    # transcript, submit an empty answer and move on. We only do this
    # once per (comp_idx, phase) by checking autoadvanced_key.
    deadline = st.session_state.get(deadline_key)
    if (
        deadline is not None
        and time.time() >= deadline
        and not st.session_state.get(transcript_shown_key)
        and not st.session_state.get(autoadvanced_key)
    ):
        # The recording_cap JS will already have clicked stop if the
        # candidate was actively recording. If a recording actually
        # came in, transcription would have flipped transcript_shown_key
        # to True before we got here (in which case we wouldn't be in
        # this branch). Treat the answer as empty and advance.
        st.session_state[autoadvanced_key] = True
        st.warning("⏰ Time's up on this question. Moving on.")
        _advance_after_answer(comp, phase, transcript="")
        return

    # --- Recording UI ---
    if not st.session_state.get(transcript_shown_key):
        if MIC_AVAILABLE:
            st.markdown(f"**Record your answer** (up to {ANSWER_WINDOW_SECONDS}s):")
            audio_input_kwargs = {
                "key": f"mic_{comp_idx}_{phase}",
                "label_visibility": "collapsed",
            }
            try:
                audio_file = st.audio_input(
                    "Click the microphone to start, click again to stop.",
                    sample_rate=16000,
                    **audio_input_kwargs,
                )
            except TypeError:
                # Older Streamlit without sample_rate support.
                audio_file = st.audio_input(
                    "Click the microphone to start, click again to stop.",
                    **audio_input_kwargs,
                )

            # The recording_cap helper renders a live countdown alongside
            # the recorder and auto-clicks stop when the cap is reached.
            # We pass the remaining seconds in the window so the cap
            # respects the same overall 90s clock (rather than giving
            # the candidate a fresh 90s once they click record).
            remaining_secs = ANSWER_WINDOW_SECONDS
            if deadline is not None:
                remaining_secs = max(1, int(deadline - time.time()))
            render_recording_cap(max_seconds=remaining_secs)

            # st.audio_input keeps returning the same UploadedFile on
            # every rerun, so we dedup by (file_id, size).
            already_done_key = f"l3_transcribed_id_{comp_idx}_{phase}"
            if audio_file is not None:
                audio_bytes = audio_file.getvalue()
                fingerprint = (audio_file.file_id, len(audio_bytes)) if hasattr(audio_file, "file_id") else (id(audio_file), len(audio_bytes))
                if st.session_state.get(already_done_key) != fingerprint:
                    st.session_state[already_done_key] = fingerprint
                    st.session_state[audio_bytes_key] = audio_bytes
                    with st.spinner("Transcribing..."):
                        try:
                            transcript = transcribe_audio(audio_bytes, filename="recording.wav")
                            if not transcript:
                                raise ValueError("Empty transcript — recording may have been silent.")
                            st.session_state[transcript_key] = transcript
                            st.session_state[transcript_shown_key] = True
                            st.rerun()
                        except Exception as e:
                            st.error(
                                f"Transcription failed: {type(e).__name__} — {e}\n\n"
                                "Type your answer below as a fallback. "
                                "If this keeps happening, check the Streamlit logs "
                                "for the underlying Azure error."
                            )

        with st.expander("Or type your answer instead"):
            typed = st.text_area(
                "Type your answer",
                key=f"typed_{comp_idx}_{phase}",
                height=180,
            )
            if st.button("Submit typed answer", key=f"submit_typed_{comp_idx}_{phase}"):
                if typed.strip():
                    st.session_state[transcript_key] = typed.strip()
                    st.session_state[transcript_shown_key] = True
                    st.rerun()
                else:
                    st.warning("Please enter an answer first.")

    # --- Review and continue ---
    else:
        transcript = st.session_state.get(transcript_key, "")
        st.markdown("**Your transcribed answer:**")
        st.write(f"> {transcript}")

        if st.button("Re-record this answer", key=f"rerecord_{comp_idx}_{phase}"):
            # Re-recording resets the per-phase transcript and audio. The
            # answer-window deadline stays anchored to the original
            # question playback, so if it's already passed re-recording
            # won't help, but we still let them try (e.g. for typed
            # fallback before auto-advance fires).
            st.session_state[transcript_shown_key] = False
            st.session_state.pop(transcript_key, None)
            st.session_state.pop(audio_bytes_key, None)
            st.rerun()

        if st.button("Continue", type="primary", key=f"continue_{comp_idx}_{phase}"):
            _advance_after_answer(comp, phase, transcript)


def _advance_after_answer(comp: dict, phase: str, transcript: str) -> None:
    """Branch on whether we just got a main answer or a follow-up answer."""
    comp_idx = st.session_state.l3_question_idx

    if phase == "main":
        # Stash the main transcript, generate a follow-up, move to followup phase.
        st.session_state[f"l3_main_transcript_{comp_idx}"] = transcript
        with st.spinner("Generating a follow-up question..."):
            followup = generate_followup(
                main_question=comp["question"],
                transcript=transcript,
                competency_name=comp["competency_name"],
                followup_goal=comp["followup_goal"],
            )
        st.session_state.l3_current_followup = followup
        st.session_state.l3_phase = "followup"
        st.session_state.l3_question_started_at = time.time()
        st.rerun()
        return

    # phase == "followup": we have everything needed to score this competency.
    main_transcript = st.session_state.get(f"l3_main_transcript_{comp_idx}", "")
    followup = st.session_state.get("l3_current_followup") or {}

    with st.spinner("Scoring this competency..."):
        result = score_competency(
            main_question=comp["question"],
            main_transcript=main_transcript,
            followup_question=followup.get("question", ""),
            followup_transcript=transcript,
            competency_name=comp["competency_name"],
            followup_goal=comp["followup_goal"],
        )

    main_dur = min(120.0, len(main_transcript.split()) / 2.5) if main_transcript else 0.0
    fu_dur = min(120.0, len(transcript.split()) / 2.5) if transcript else 0.0

    db.save_layer3_result(
        candidate_id=st.session_state.candidate_id,
        competency_order=comp_idx + 1,
        competency_id=comp["competency_id"],
        competency_key=comp["competency_key"],
        competency_name=comp["competency_name"],
        main_question=comp["question"],
        main_transcript=main_transcript,
        main_audio_duration_seconds=main_dur,
        followup_bucket=followup.get("bucket"),
        followup_question=followup.get("question"),
        followup_transcript=transcript,
        followup_audio_duration_seconds=fu_dur,
        competency_score=result["score"],
        scripted_flag=result["scripted_flag"],
        rationale=result["rationale"],
    )

    st.session_state.l3_answer_scores.append({
        "competency_key": comp["competency_key"],
        "competency_id": comp["competency_id"],
        "score": result["score"],
        "scripted_flag": result["scripted_flag"],
    })

    # Advance to next competency.
    st.session_state.l3_question_idx = comp_idx + 1
    st.session_state.l3_phase = "main"
    st.session_state.l3_current_followup = None
    st.session_state.l3_question_started_at = time.time()
    st.rerun()


def _finish_layer() -> None:
    st.title("Layer 3 Complete")
    st.success(
        "You've completed all three layers. On the next screen you'll see your "
        "full results and personalized feedback."
    )

    if st.button("See my results", type="primary", use_container_width=True):
        advance_stage("results")
