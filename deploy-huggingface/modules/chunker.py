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
    segments: List[Dict], max_words_per_chunk: int = None
) -> List[Dict]:
    """
    Adaptive time-aligned chunking for subtitle segments (SRT, VTT, Whisper ASR).
    - For short videos (<= 320 words, ~1-2 min): splits into 2 balanced chunks so both
      the problem statement and the solution/conclusion are preserved in full detail.
    - For medium/long videos (> 320 words): splits into 4-10 balanced thematic chapters
      (120-220 words per chapter), optimized for both deep coverage and CPU inference speed.
    - Any trailing fragment under 40 words is merged into the previous chapter.
    """
    if not segments:
        return []

    # Calculate total words across all segments
    total_words = sum(len(seg.get("text", "").split()) for seg in segments)

    if max_words_per_chunk is not None and max_words_per_chunk > 0:
        target_words = max_words_per_chunk
    elif total_words <= 320:
        target_words = max(60, total_words // 2)
    elif total_words <= 650:
        target_words = max(180, total_words // 2)
    else:
        target_words = min(260, max(180, total_words // 5))

    chunks = []
    current_words = []
    current_start = segments[0].get("start", 0.0)
    current_end = segments[0].get("end", 0.0)

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        words = text.split()
        if len(current_words) + len(words) > target_words and current_words:
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
        # Merge trailing tiny fragment (< 40 words) into previous chunk if possible
        if len(current_words) < 40 and chunks:
            chunks[-1]["text"] += " " + " ".join(current_words)
            chunks[-1]["end_time"] = current_end
            chunks[-1]["time_str"] = f"{format_seconds_to_time(chunks[-1]['start_time'])} - {format_seconds_to_time(current_end)}"
            chunks[-1]["word_count"] += len(current_words)
        else:
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
    text: str, max_words_per_chunk: int = None
) -> List[Dict]:
    """
    Adaptive semantic chunking for direct text / transcript.
    - Texts < 100 words (or 1-2 sentences): preserved as a single cohesive unit.
    - Texts 100 - 220 words with >= 3 sentences: partitioned into 2 balanced thematic sections
      (e.g., Problem & Context vs. Solutions & Recommendations), preventing Seq2Seq lead/copy bias.
    - Texts 220 - 360 words: partitioned into 3 balanced thematic sections.
    - Texts > 360 words: partitioned into balanced thematic sections (~120 words each).
    """
    text = (text or "").strip()
    if not text:
        return []

    words = text.split()
    total_words = len(words)

    # Sentences split
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…\n])\s+", text) if s.strip()]

    # Very short text or single sentence: keep intact
    if total_words < 100 or len(sentences) <= 2:
        return [{
            "chunk_id": 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": "Toàn văn",
            "text": text,
            "word_count": total_words,
        }]

    # Paragraph split if clear structured paragraphs exist
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1 and all(len(p.split()) >= 40 for p in paragraphs):
        chunks = []
        for i, p in enumerate(paragraphs):
            chunks.append({
                "chunk_id": i + 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": f"Phần {i + 1}",
                "text": p,
                "word_count": len(p.split()),
            })
        return chunks

    # Determine optimal number of chunks
    if max_words_per_chunk is not None and max_words_per_chunk > 0:
        num_chunks = max(2, round(total_words / max_words_per_chunk))
    elif total_words <= 220:
        num_chunks = 2
    elif total_words <= 360:
        num_chunks = 3
    else:
        num_chunks = max(3, round(total_words / 120))

    target_words_per_chunk = total_words / num_chunks

    chunks = []
    cur_sentences = []
    cur_word_count = 0

    thematic_titles_vi = [
        "Phần 1: Bối cảnh & Ứng dụng thực tiễn",
        "Phần 2: Thách thức & Đánh giá phân tích",
        "Phần 3: Giải pháp & Khuyến nghị tương lai",
        "Phần 4: Kết luận & Tầm nhìn chiến lược"
    ]

    for s in sentences:
        s_words = len(s.split())
        if cur_word_count + s_words > target_words_per_chunk and len(chunks) < num_chunks - 1 and cur_sentences:
            idx = len(chunks)
            title = thematic_titles_vi[idx] if idx < len(thematic_titles_vi) else f"Phần {idx + 1}"
            chunks.append({
                "chunk_id": idx + 1,
                "start_time": 0.0,
                "end_time": 0.0,
                "time_str": title,
                "text": " ".join(cur_sentences),
                "word_count": cur_word_count,
            })
            cur_sentences = [s]
            cur_word_count = s_words
        else:
            cur_sentences.append(s)
            cur_word_count += s_words

    if cur_sentences:
        idx = len(chunks)
        title = thematic_titles_vi[idx] if idx < len(thematic_titles_vi) else f"Phần {idx + 1}"
        chunks.append({
            "chunk_id": idx + 1,
            "start_time": 0.0,
            "end_time": 0.0,
            "time_str": title,
            "text": " ".join(cur_sentences),
            "word_count": cur_word_count,
        })

    return chunks
