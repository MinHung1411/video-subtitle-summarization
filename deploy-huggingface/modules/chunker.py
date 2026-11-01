"""
Long-context Chunking Module:
Splits transcripts and plain texts into balanced thematic sections
so that Seq2Seq summarization captures all key ideas (opening, body, conclusion)
without dropping earlier sentences or hallucinating.
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
    Thematic section chunking for direct text / transcript.
    Splits multi-sentence text into logical thematic sections (opening, middle, conclusion)
    so that summarization captures all key ideas and never drops early paragraphs.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…\n])\s+", text) if s.strip()]
    n = len(sentences)
    words = text.split()
    total_words = len(words)

    # Very short texts (1-2 sentences or <= 45 words): 1 single chunk
    if n <= 2 or total_words <= 45:
        return [{
            "chunk_id": 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": "Toàn văn",
            "text": text,
            "word_count": total_words,
        }]

    # Short to medium texts (3 to 6 sentences, e.g. 50-250 words like sample texts):
    # Split into 2 balanced thematic sections: First half (Opening & Core points), Second half (Challenges & Conclusion)
    if n <= 6 and total_words <= 260:
        mid = max(1, n // 2)
        part1_text = " ".join(sentences[:mid])
        part2_text = " ".join(sentences[mid:])
        return [
            {
                "chunk_id": 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": "Phần 1: Bối cảnh & Ứng dụng",
                "text": part1_text,
                "word_count": len(part1_text.split()),
            },
            {
                "chunk_id": 2,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": "Phần 2: Thách thức & Khuyến nghị",
                "text": part2_text,
                "word_count": len(part2_text.split()),
            }
        ]

    # Medium texts (7 to 12 sentences, e.g. 250-450 words):
    # Split into 3 sections: Opening, Deep Dive / Evidence, Conclusion
    if n <= 12 and total_words <= 450:
        p1 = max(1, n // 3)
        p2 = max(p1 + 1, 2 * n // 3)
        part1 = " ".join(sentences[:p1])
        part2 = " ".join(sentences[p1:p2])
        part3 = " ".join(sentences[p2:])
        return [
            {
                "chunk_id": 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": "Phần 1: Khởi đầu & Bối cảnh",
                "text": part1,
                "word_count": len(part1.split()),
            },
            {
                "chunk_id": 2,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": "Phần 2: Phân tích & Thực trạng",
                "text": part2,
                "word_count": len(part2.split()),
            },
            {
                "chunk_id": 3,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": "Phần 3: Đánh giá & Khuyến nghị",
                "text": part3,
                "word_count": len(part3.split()),
            }
        ]

    # Longer texts (> 450 words): split by sentence boundaries respecting max_words_per_chunk
    chunks = []
    cur_sentences = []
    cur_word_count = 0

    for s in sentences:
        s_words = len(s.split())
        if cur_word_count + s_words > max_words_per_chunk and cur_sentences:
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": f"Chương {len(chunks) + 1}",
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
            "time_str": f"Chương {len(chunks) + 1}",
            "text": " ".join(cur_sentences),
            "word_count": cur_word_count,
        })

    return chunks
