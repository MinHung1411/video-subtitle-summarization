"""
Long-context Chunking Module:
Splits long transcripts into semantic chunks to respect Transformer context limits
with adaptive sentence-boundary splitting for rich multi-perspective summarization.
"""

import re
from typing import List, Dict
from .audio_handler import format_seconds_to_time

def chunk_by_segments(
    segments: List[Dict], max_words_per_chunk: int = 280
) -> List[Dict]:
    """
    Time-aligned chunking for subtitle segments (SRT, VTT, Whisper ASR).
    Keeps chronological start/end timestamps and splits on segment boundaries.
    """
    if not segments:
        return []

    chunks = []
    current_words = []
    current_start = segments[0].get("start", 0.0)
    current_end = segments[0].get("end", 0.0)

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
    text: str, max_words_per_chunk: int = 280
) -> List[Dict]:
    """
    Adaptive sentence-aware chunking for direct text / transcript.
    Intelligently splits short and medium texts into logical sections
    so that Seq2Seq models produce comprehensive, multi-paragraph summaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    total_words = len(words)

    # For very short texts (<= 55 words), keep as 1 chunk
    if total_words <= 55:
        return [{
            "chunk_id": 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": "Toàn văn",
            "text": text,
            "word_count": total_words,
        }]

    # Split by sentence boundaries (preserving Vietnamese and punctuation)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…\n])\s+", text) if s.strip()]
    if not sentences or len(sentences) == 1:
        return [{
            "chunk_id": 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": "Toàn văn",
            "text": text,
            "word_count": total_words,
        }]

    # Determine adaptive target chunk size based on text length
    if total_words <= 150:
        target_chunk_size = 60
    elif total_words <= 320:
        target_chunk_size = 110
    else:
        target_chunk_size = max_words_per_chunk

    chunks = []
    cur_sentences = []
    cur_word_count = 0

    for s in sentences:
        s_words = len(s.split())
        if cur_word_count + s_words > target_chunk_size and cur_sentences:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": f"Phân đoạn {len(chunks) + 1}",
                "text": " ".join(cur_sentences),
                "word_count": cur_word_count,
            })
            cur_sentences = [s]
            cur_word_count = s_words
        else:
            cur_sentences.append(s)
            cur_word_count += s_words

    if cur_sentences:
        chunks.append({
            "chunk_id": len(chunks) + 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": f"Phân đoạn {len(chunks) + 1}",
            "text": " ".join(cur_sentences),
            "word_count": cur_word_count,
        })

    return chunks
