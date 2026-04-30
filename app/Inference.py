"""
Unified transcription using faster-whisper.
Handles both batch (full audio file) and live (short numpy buffer).
Silero VAD built in via vad_filter=True.
Optional LLM post-correction via Claude (batch only).
"""
import os
import json
import numpy as np
from faster_whisper import WhisperModel
import httpx

MODEL_PATH = os.getenv("MODEL_PATH_CT2", "/app/model-ct2")

# LLM correction settings (batch only)
LLM_CORRECTION_ENABLED = os.getenv("LLM_CORRECTION_ENABLED", "true").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5")
LLM_WINDOW_SIZE = int(os.getenv("LLM_WINDOW_SIZE", "15"))   # segments per LLM call
LLM_WINDOW_OVERLAP = int(os.getenv("LLM_WINDOW_OVERLAP", "2"))  # overlap on both sides

_model: WhisperModel | None = None


def _strip_bom_if_present(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        data = f.read()
    if data.startswith(b"\xef\xbb\xbf"):
        with open(path, "wb") as f:
            f.write(data[3:])
        print(f"  [INFERENCE] Stripped BOM from {path}")


def load_model():
    global _model
    if _model is not None:
        return
    _strip_bom_if_present(os.path.join(MODEL_PATH, "preprocessor_config.json"))
    _strip_bom_if_present(os.path.join(MODEL_PATH, "config.json"))
    print(f"  [INFERENCE] Loading faster-whisper from {MODEL_PATH}...")
    _model = WhisperModel(
        MODEL_PATH,
        device="cuda",
        compute_type="float16",
    )
    print("  [INFERENCE] Model loaded.")


def load_model_live():
    load_model()


# ─────────────────────────────────────────────────────────────────────────────
# LLM post-correction
# ─────────────────────────────────────────────────────────────────────────────

CORRECTION_SYSTEM_PROMPT = """You are an Arabic ASR transcript corrector specialized in Levantine dialect (Lebanese, Syrian, Palestinian, Jordanian).

You will receive numbered transcript segments. For EACH segment, return a corrected version.

RULES:
1. PRESERVE the Levantine dialect. Do NOT convert to Modern Standard Arabic.
   - Keep "هلق", "بدي", "شو", "هلكي", "بدها", "إيه", "كيفك" etc. as-is.
   - Keep "هلق" — do NOT change to "الآن"
   - Keep "بدي" — do NOT change to "أريد"
   - Keep "شو" — do NOT change to "ماذا"
2. Fix obvious spelling errors that change meaning (e.g. "الكرامي" → "الكرامة", "بشدي" → "بشدة").
3. Fix words that are stuck together OR incorrectly split (e.g. "بسي حة" → "بسيحة").
4. Add minimal punctuation: periods, commas, question marks where natural.
5. Do NOT merge or split segments. Each input number gets exactly one output.
6. Do NOT change segment numbers or timing.
7. If a segment is already correct, return it unchanged.
8. If a segment is genuinely garbled and you can't recover meaning, return it unchanged rather than guessing.
9. Output ONLY a JSON object: {"corrections": [{"id": <int>, "text": "<corrected>"}, ...]}
   No prose, no explanation, no markdown fences."""


async def _call_claude(segments_window: list[dict]) -> dict[int, str]:
    """
    Send a window of segments to Claude, return {seg_id: corrected_text}.
    """
    if not ANTHROPIC_API_KEY:
        return {}

    user_content = "Correct these Levantine Arabic transcript segments:\n\n"
    for seg in segments_window:
        user_content += f"{seg['id']}. {seg['text']}\n"

    payload = {
        "model": LLM_MODEL,
        "max_tokens": 4096,
        "system": CORRECTION_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["content"][0]["text"].strip()
        # Strip markdown fences if model adds them despite the prompt
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        return {item["id"]: item["text"] for item in parsed.get("corrections", [])}
    except Exception as e:
        print(f"  [LLM] correction call failed: {e}")
        return {}


async def _correct_with_llm(segments: list[dict]) -> list[dict]:
    """
    Run sliding-window LLM correction over all segments.
    Returns a new segments list with text replaced where the LLM responded.
    """
    if not segments or not ANTHROPIC_API_KEY:
        return segments

    print(f"  [LLM] Correcting {len(segments)} segments with {LLM_MODEL}...")

    # Assign stable ids
    indexed = [{"id": i, "text": s["text"], "start": s["start"], "end": s["end"]}
               for i, s in enumerate(segments)]

    corrected_text: dict[int, str] = {}
    step = max(1, LLM_WINDOW_SIZE - LLM_WINDOW_OVERLAP)
    i = 0
    while i < len(indexed):
        window = indexed[i : i + LLM_WINDOW_SIZE]
        result = await _call_claude(window)
        # Only overwrite when not already set (so center segments win over edges)
        for seg_id, txt in result.items():
            if seg_id not in corrected_text:
                corrected_text[seg_id] = txt
        i += step

    out = []
    for seg in indexed:
        new_text = corrected_text.get(seg["id"], seg["text"])
        out.append({"start": seg["start"], "end": seg["end"], "text": new_text.strip()})

    fixed = sum(1 for s, o in zip(segments, out) if s["text"] != o["text"])
    print(f"  [LLM] Done. Modified {fixed}/{len(segments)} segments.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

# Levantine drama-domain prompt — biases the decoder toward dialect spellings
# and away from MSA. Keep it short; long prompts hurt accuracy.
LEVANTINE_PROMPT = (
    "حوار باللهجة العامية الشامية. هلق، بدي، شو، كيفك، هلكي، إيه، لك، يلا، هيك."
)


async def transcribe(audio, dialect="auto", initial_prompt=None, apply_llm_correction=None):
    """
    Accepts either:
      - str path to a WAV/audio file (batch)
      - np.ndarray float32 mono 16kHz audio (live)
    Returns a dict with 'text' and 'segments' for batch,
    or a plain str of joined text for live.

    apply_llm_correction:
      - None  -> use LLM_CORRECTION_ENABLED env default (batch only)
      - True  -> force on  (batch only; ignored for live)
      - False -> force off
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    is_live = isinstance(audio, np.ndarray)

    # Pick a sensible default prompt for Levantine if caller didn't supply one
    prompt = initial_prompt if initial_prompt is not None else LEVANTINE_PROMPT

    segments, info = _model.transcribe(
        audio,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        # ── anti-repetition / anti-loop / anti-hallucination tuning ──
        condition_on_previous_text=False,        # kills chorus loops
        no_repeat_ngram_size=3,                  # no exact 3-gram repeats per segment
        repetition_penalty=1.4,                  # discourage repeated tokens
        compression_ratio_threshold=2.2,         # tighter; flags repetitive output
        log_prob_threshold=-1.0,                 # discard low-confidence segments
        no_speech_threshold=0.6,                 # require clearer speech to emit
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8],   # fallback ladder on failures
        word_timestamps=True,
        initial_prompt=prompt,
    )

    seg_list = []
    text_parts = []
    for seg in segments:
        t = seg.text.strip()
        if not t:
            continue
        seg_list.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": t,
        })
        text_parts.append(t)

    # Live path — no LLM correction (latency), no structured output
    if is_live:
        return " ".join(text_parts).strip()

    # Batch path — optional LLM correction
    do_correction = (
        LLM_CORRECTION_ENABLED if apply_llm_correction is None else apply_llm_correction
    )
    if do_correction and seg_list:
        seg_list = await _correct_with_llm(seg_list)
        text_parts = [s["text"] for s in seg_list]

    return {
        "text": " ".join(text_parts).strip(),
        "segments": seg_list,
        "duration_seconds": float(info.duration),
        "language": info.language,
    }


def save_result(result: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)