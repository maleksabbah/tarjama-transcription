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
LLM_WINDOW_SIZE = int(os.getenv("LLM_WINDOW_SIZE", "15"))
LLM_WINDOW_OVERLAP = int(os.getenv("LLM_WINDOW_OVERLAP", "2"))

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

CORRECTION_SYSTEM_PROMPT = """أنت مصحح نصوص ASR متخصص في اللهجة الشامية (لبنان، سوريا، فلسطين، الأردن).

ستستلم مقاطع نص مرقمة من تفريغ صوتي. صحح كل مقطع.

قواعد إلزامية:
1. حافظ على اللهجة الشامية. ممنوع التحويل للفصحى.
   - "هلق" تبقى "هلق" (لا "الآن")
   - "بدي" تبقى "بدي" (لا "أريد")
   - "شو" تبقى "شو" (لا "ماذا")
   - "كيفك" تبقى "كيفك"
   - "هيك" تبقى "هيك"

2. صحح الأخطاء الإملائية الواضحة، خصوصاً:
   - الكلمات المقطوعة في غير محلها: "اشيا" → "أشياء"، "بسيحة" تبقى "بسيحة"
   - الكلمات المدمجة خطأً: "ليقات" → "لي قات"، "هلقوقف" → "هلق وقف"
   - الهمزات الناقصة: "اسمعت" → "أسمعت"، "اني" → "إني"
   - أخطاء التعرف الصوتي الشائعة:
     • "الكرامي" → "الكرامة"
     • "اغساك" → "أقسى"
     • "بشدي" → "بشدة"
     • "نقاد" → "نقاط"
     • "اني عالى" → "أنا علي"
     • "اهرا وسهلك" → "أهلاً وسهلاً"
     • "اشكال الوان" → "أشكال ألوان"
     • "زلمنا" → "زلمنا" (تبقى)
     • "مسخني" → "مش هني" أو حسب السياق

3. أضف علامات ترقيم بسيطة (نقطة، فاصلة، علامة استفهام) عند اللزوم.

4. لا تدمج المقاطع ولا تقسمها. كل رقم input له output واحد بنفس الرقم.

5. إذا المقطع مفهوم وصحيح، أعده كما هو.

6. إذا المقطع مدمر تماماً ولا يمكن استرداد المعنى، أعده كما هو بدون تخمين.

7. أرجع JSON فقط، بدون أي شرح أو نص إضافي:
{"corrections": [{"id": <int>, "text": "<corrected>"}, ...]}

لا تكتب markdown fences. لا تشرح. JSON فقط."""


async def _call_claude(segments_window: list[dict]) -> dict[int, str]:
    if not ANTHROPIC_API_KEY:
        return {}

    user_content = "صحح هذه المقاطع من تفريغ صوتي بالعامية الشامية:\n\n"
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
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(text)
        return {item["id"]: item["text"] for item in parsed.get("corrections", [])}
    except Exception as e:
        print(f"  [LLM] correction call failed: {e}")
        return {}


async def _correct_with_llm(segments: list[dict]) -> list[dict]:
    if not segments or not ANTHROPIC_API_KEY:
        return segments

    print(f"  [LLM] Correcting {len(segments)} segments with {LLM_MODEL}...")

    indexed = [{"id": i, "text": s["text"], "start": s["start"], "end": s["end"]}
               for i, s in enumerate(segments)]

    corrected_text: dict[int, str] = {}
    step = max(1, LLM_WINDOW_SIZE - LLM_WINDOW_OVERLAP)
    i = 0
    while i < len(indexed):
        window = indexed[i : i + LLM_WINDOW_SIZE]
        result = await _call_claude(window)
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

LEVANTINE_PROMPT = (
    "حوار باللهجة العامية الشامية. هلق، بدي، شو، كيفك، هلكي، إيه، لك، يلا، هيك."
)


async def transcribe(audio, dialect="auto", initial_prompt=None, apply_llm_correction=None):
    """
    Accepts either:
      - str path to a WAV/audio file (batch)
      - np.ndarray float32 mono 16kHz audio (live)
    Returns dict for batch, plain str for live.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    is_live = isinstance(audio, np.ndarray)
    prompt = initial_prompt if initial_prompt is not None else LEVANTINE_PROMPT

    segments, info = _model.transcribe(
        audio,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        # ── conservative anti-loop tuning ──
        condition_on_previous_text=False,        # KEY fix: stops chorus loops
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,                 # gentle, not aggressive
        compression_ratio_threshold=2.4,         # whisper default
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8],   # fallback ladder
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

    if is_live:
        return " ".join(text_parts).strip()

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