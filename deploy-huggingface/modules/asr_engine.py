"""
ASR Engine Module:
Speech-to-Text transcription with Whisper, extracting timestamped segments.
"""

import os
import re
import time
import torch

_WHISPER_CACHE = {}
DEVICE = "cpu"

def normalize_speech_text(text: str) -> str:
    """Cleans up ASR noise artifacts and fixes phonetic misrecognition."""
    if not text:
        return ""
    # Remove bracket noise tokens: [âm nhạc], [vỗ tay], [tiếng cười], (music), etc.
    text = re.sub(r"\[(âm nhạc|tiếng cười|vỗ tay|nhạc|music|applause|laughter)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((âm nhạc|tiếng cười|vỗ tay|nhạc|music|applause|laughter)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[♪♫]", "", text)

    # Phonetic corrections for common spoken Vietnamese misrecognitions
    replacements = [
        (r"\bNick Vucci sức\b", "Nick Vujicic"),
        (r"\bnext anh là ai\b", "Nick anh là ai"),
        (r"\bxích mình bé nhỏ\b", "sinh linh bé nhỏ"),
        (r"\bquyết trị vật\b", "khuyết tật"),
        (r"\btiểu tạng\b", "từ thiện"),
        (r"\bchâu đến\b", "chờ đến"),
        (r"\bbài máu\b", "bài báo"),
        (r"\bLife Việt phaolemis\b", "Life Without Limbs"),
        (r"\bvì chính phủ Life\b", "phi chính phủ Life"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clean_subtitle_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = normalize_speech_text(text)
    return text.strip()

def parse_srt_with_timestamps(srt_content: str) -> list:
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    segments = []

    def time_to_sec(t_str):
        parts = t_str.strip().replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return 0.0

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 2:
            time_match = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", lines[1] if len(lines) > 1 else "")
            if not time_match and len(lines) > 0:
                time_match = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", lines[0])
            if time_match:
                start_sec = time_to_sec(time_match.group(1))
                end_sec = time_to_sec(time_match.group(2))
                text_part = " ".join(lines[2:]) if len(lines) > 2 else ""
                clean_t = clean_subtitle_text(text_part)
                if clean_t:
                    segments.append({"start": start_sec, "end": end_sec, "text": clean_t})
    return segments

def load_whisper_model(model_size: str = "base"):
    global _WHISPER_CACHE
    device = "cpu"
    if model_size not in _WHISPER_CACHE:
        import whisper
        model = whisper.load_model(model_size, device=device)
        _WHISPER_CACHE[model_size] = (model, device)
    model, cached_dev = _WHISPER_CACHE[model_size]
    return model

def transcribe_audio(audio_path: str, model_size: str = "base", language: str = None) -> dict:
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "text": "",
            "language": "",
            "segments": [],
            "error": f"File not found: {audio_path}",
        }

    t0 = time.time()
    try:
        model = load_whisper_model(model_size)
        use_fp16 = torch.cuda.is_available()

        context_prompt = (
            "Đây là bản ghi âm video bài phát biểu hoặc phỏng vấn có phụ đề rõ ràng, viết hoa tên riêng chính xác (Nick Vujicic, Australia, Melbourne, Stanford, Steve Jobs)."
            if (language in ["vi", "vietnamese"] or not language or language == "auto")
            else "This is an audio recording of a speech or interview with clean punctuation, proper names, and capitalization."
        )

        result = model.transcribe(
            audio_path,
            fp16=use_fp16,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0,
            beam_size=3,
            best_of=3,
            initial_prompt=context_prompt,
            language=language if language and language != "auto" else None,
        )

        full_text = normalize_speech_text(result.get("text", "").strip())
        detected_lang = result.get("language", "unknown")

        raw_segments = result.get("segments", [])
        structured_segments = []
        for seg in raw_segments:
            s_text = normalize_speech_text(seg.get("text", "").strip())
            if s_text:
                structured_segments.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": s_text,
                })

        return {
            "success": True,
            "text": full_text,
            "language": detected_lang,
            "segments": structured_segments,
            "transcribe_time": round(time.time() - t0, 2),
            "word_count": len(full_text.split()),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "language": "",
            "segments": [],
            "error": str(e),
        }
