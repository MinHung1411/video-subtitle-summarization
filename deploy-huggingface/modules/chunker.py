"""
Long-context Chunking Module:
Splits long transcripts into semantic chunks to respect Transformer context limits.
"""

from typing import List, Dict
from .audio_handler import format_seconds_to_time

def chunk_by_segments(
    segments: List[Dict], max_words_per_chunk: int = 400
) -> List[Dict]:
    if not segments:
        return []

    chunks = []
    current_words = []
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        words = text.split()
        if len(current_words) + len(words) > max_words_per_chunk and current_words:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "start_time": current_start,
                "end_time": current_end,
                "time_str": f"{format_seconds_to_time(current_start)} - {format_seconds_to_time(current_end)}",
                "text": " ".join(current_words),
                "word_count": len(current_words),
            })
            current_words = list(words)
            current_start = seg.get("start", current_end)
            current_end = seg.get("end", current_start)
        else:
            current_words.extend(words)
            current_end = seg.get("end", current_end)

    if current_words:
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_time": current_start,
            "end_time": current_end,
            "time_str": f"{format_seconds_to_time(current_start)} - {format_seconds_to_time(current_end)}",
            "text": " ".join(current_words),
            "word_count": len(current_words),
        })

    return chunks

def chunk_plain_text(
    text: str, max_words_per_chunk: int = 400, overlap_words: int = 50
) -> List[Dict]:
    words = text.split()
    if len(words) <= max_words_per_chunk:
        return [{
            "chunk_id": 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": "Full",
            "text": text,
            "word_count": len(words),
        }]

    chunks = []
    start_idx = 0
    step = max_words_per_chunk - overlap_words

    while start_idx < len(words):
        chunk_words = words[start_idx : start_idx + max_words_per_chunk]
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": f"Part {len(chunks) + 1}",
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words),
        })
        start_idx += step
        if start_idx >= len(words):
            break

    return chunks
