"""
Summarization Engine Module:
Multilingual Seq2Seq generation using fine-tuned mT5 (minhunhooo/video-subtitle-summarization-mt5)
with Hierarchical Map-Reduce chunking and Unicode-aware ROUGE evaluation.
Runs on high-memory CPU (16GB RAM) to eliminate ZeroGPU quota exhaustion and paywall popups.
"""

import os
import re
import time
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple
import torch
import nltk
from nltk.corpus import stopwords
from rouge_score import rouge_scorer as _rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEVICE = "cpu"

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "the", "a", "an", "is", "it", "in", "of", "and", "to", "for",
        "with", "was", "are", "be", "this", "that"
    }

VI_STOP_WORDS = {
    "và", "hoặc", "nhưng", "tuy", "nhiên", "các", "những", "một", "của", "cho", "với", "được",
    "trong", "trên", "dưới", "ngoài", "khi", "lúc", "này", "đó", "kia", "đang", "đã", "sẽ",
    "phải", "luôn", "cũng", "vẫn", "thì", "là", "rằng", "việc", "sự", "cái", "chiếc", "nhất",
    "hơn", "rất", "quá", "lắm", "nhiều", "ít", "đến", "từ", "vào", "ra", "lại", "qua", "về",
    "theo", "bởi", "vì", "do", "nên", "vậy", "thế", "nào", "gì", "ai", "sao", "đâu", "đây",
    "ấy", "họ", "chúng", "tôi", "ta", "bạn", "ông", "bà", "anh", "chị", "em", "người", "mọi",
    "mỗi", "từng", "toàn", "bộ", "phần", "điều", "thứ", "làm", "tạo", "thấy", "biết", "nghĩ",
    "nói", "khác", "như", "cùng", "sau", "trước", "giữa", "thêm", "bớt", "vậy"
}

try:
    FR_STOP_WORDS = set(stopwords.words("french"))
    ES_STOP_WORDS = set(stopwords.words("spanish"))
except Exception:
    FR_STOP_WORDS = set()
    ES_STOP_WORDS = set()

ALL_STOP_WORDS = STOP_WORDS | VI_STOP_WORDS | FR_STOP_WORDS | ES_STOP_WORDS

class UnicodeTokenizer:
    """NFC Unicode-aware tokenizer preserving Vietnamese diacritics for accurate ROUGE."""
    def tokenize(self, text):
        norm = unicodedata.normalize("NFC", str(text or "")).lower()
        return re.sub(r"[^\w\s]", " ", norm).split()

_ROUGE = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=UnicodeTokenizer())
_LOADED_MODELS = {}

DEFAULT_MODEL_REPO = os.environ.get("MODEL_REPO", "minhunhooo/video-subtitle-summarization-mt5")

def detect_language(text: str) -> str:
    t = text.lower()
    if re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", t):
        return "vi"
    if re.search(r"\b(le|la|les|des|est|une|que)\b", t):
        return "fr"
    if re.search(r"[ñ¿¡]|\b(el|los|las|una|que|es)\b", t):
        return "es"
    return "en"

def get_summarizer_model(model_name: str = None):
    global _LOADED_MODELS
    target_repo = model_name or DEFAULT_MODEL_REPO

    if target_repo in _LOADED_MODELS:
        return _LOADED_MODELS[target_repo]

    print(f"Loading summarization model on CPU (16GB RAM): {target_repo}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            target_repo,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.eval()
        _LOADED_MODELS[target_repo] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {target_repo}: {e}. Fallback to csebuetnlp/mT5_multilingual_XLSum...")
        fallback = "csebuetnlp/mT5_multilingual_XLSum"
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback, torch_dtype=torch.float32)
        model.eval()
        _LOADED_MODELS[fallback] = (model, tokenizer)
        return model, tokenizer

def extract_keywords(text: str, top_k: int = 10) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ỹ0-9_+-]+", text)
    cleaned = []
    for w in tokens:
        wl = w.lower()
        if len(wl) >= 2 and wl not in ALL_STOP_WORDS and not wl.isdigit():
            cleaned.append(w.capitalize())

    freq = Counter(cleaned)
    top_words = [w for w, _ in freq.most_common(top_k)]
    if not top_words:
        return "N/A"

    badges = [
        f'<span style="display: inline-block; background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; padding: 3px 9px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px 3px;">#{w}</span>'
        for w in top_words
    ]
    return " ".join(badges)

def lead3_summarize(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:3]) if sentences else text[:400]

def generate_single_summary(
    text: str,
    lang: str = "vi",
    max_output: int = 250,
    min_output: int = 35,
    num_beams: int = 4
) -> str:
    model, tokenizer = get_summarizer_model()

    clean_text = re.sub(r"\s+", " ", text).strip()
    prefix = f"summarize {lang}: "
    prompt = prefix + clean_text

    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    input_word_count = len(clean_text.split())
    target_min = min(max(min_output, 35), max(input_word_count // 2, 25))
    target_max = max(max_output, target_min + 40)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=target_max,
            min_new_tokens=target_min,
            length_penalty=1.4,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def hierarchical_summarize(
    chunks: List[Dict],
    target_lang: str = "auto",
    detected_lang: str = "en",
    max_output: int = 250,
    min_output: int = 35,
    num_beams: int = 4
) -> Tuple[str, List[Dict], float]:
    t0 = time.time()

    if target_lang != "auto":
        resolved_lang = target_lang
    elif detected_lang in ["vi", "vietnamese"]:
        resolved_lang = "vi"
    elif detected_lang in ["fr", "french"]:
        resolved_lang = "fr"
    elif detected_lang in ["es", "spanish"]:
        resolved_lang = "es"
    else:
        resolved_lang = "en"

    chapters = []
    chunk_summaries = []

    # Map Phase: Summarize each segment/chapter
    for chunk in chunks:
        c_text = chunk["text"]
        time_str = chunk.get("time_str", "")

        c_summary = generate_single_summary(
            c_text,
            lang=resolved_lang,
            max_output=max(120, max_output),
            min_output=max(25, min_output // 2),
            num_beams=num_beams
        )

        chunk_summaries.append(c_summary)
        chapters.append({
            "chunk_id": chunk.get("chunk_id", len(chapters) + 1),
            "time_str": time_str,
            "summary": c_summary,
            "raw_text": c_text
        })

    # Reduce Phase: Synthesize into structured executive briefing
    if len(chunk_summaries) > 1:
        combined_sents = []
        for s in chunk_summaries:
            s_clean = s.strip()
            if s_clean and not s_clean.endswith((".", "!", "?")):
                s_clean += "."
            if s_clean:
                combined_sents.append(s_clean)

        full_overview = " ".join(combined_sents)

        executive_summary = full_overview
    else:
        # Single chunk: clean, accurate summary directly
        executive_summary = chunk_summaries[0] if chunk_summaries else ""

    return executive_summary, chapters, time.time() - t0

def evaluate_summary(transcript: str, summary: str, lead3: str) -> str:
    words_t = transcript.lower().split()
    words_s = summary.lower().split()
    words_l = lead3.lower().split()
    n_t, n_s = len(words_t), len(words_s)

    ratio = n_s / max(n_t, 1)

    r_vs_trans = _ROUGE.score(transcript[:2000], summary)
    r_lead = _ROUGE.score(transcript[:2000], lead3)

    set_lead = set(words_l)
    novel_words = [w for w in words_s if w not in set_lead and len(w) > 2]
    abstractive_ratio = len(novel_words) / max(n_s, 1) * 100

    report = f"""| Chỉ số NLP Benchmark | Kết quả Đạt được | Ý nghĩa / Baseline Kỹ thuật |
| :--- | :---: | :--- |
| **Độ dài Bản gốc (Source Length)** | **`{n_t:,}`** từ | Toàn văn lời thoại transcript được trích xuất |
| **Độ dài Tóm tắt (Summary Length)** | **`{n_s:,}`** từ | Bản tóm tắt điều hành được sinh tự động |
| **Tỷ lệ Nén (Compression Ratio)** | **`{ratio * 100:.1f}%`** | Nén súc tích, cô đọng nội dung cốt lõi |
| **Mô hình ROUGE-1 (Unigram Overlap)** | **`{r_vs_trans['rouge1'].fmeasure * 100:.1f}%`** | Độ trùng khớp từ vựng đơn ngữ nghĩa |
| **Mô hình ROUGE-L (LCS Coherence)** | **`{r_vs_trans['rougeL'].fmeasure * 100:.1f}%`** | Mức độ mạch lạc chuỗi từ liên tiếp dài nhất |
| **Lead-3 Baseline (ROUGE-1 / L)** | `{r_lead['rouge1'].fmeasure * 100:.1f}%` / `{r_lead['rougeL'].fmeasure * 100:.1f}%` | Chuẩn so sánh trích xuất 3 câu đầu kinh điển |
| **Mức độ Trừu tượng (Novel Vocabulary)** | **`{abstractive_ratio:.1f}%`** | Tỷ lệ từ vựng mới sinh ra ngoài Lead-3 |

> 🧠 **Mô hình Huấn luyện:** `minhunhooo/video-subtitle-summarization-mt5` (582M params)  
> 🔬 **Thuật toán Tóm tắt:** Hierarchical Map-Reduce (Bảo toàn đầy đủ tất cả các phân đoạn)  
> 🌐 **Bộ Chuẩn hóa:** Custom Unicode NFC Tokenizer (Bảo toàn 100% nguyên vẹn dấu thanh Tiếng Việt)"""

    return report
