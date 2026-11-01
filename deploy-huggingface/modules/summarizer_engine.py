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

def extract_keywords(text: str, top_k: int = 12) -> str:
    words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in STOP_WORDS and len(w) > 3]
    freq = Counter(words)
    top = [w for w, _ in freq.most_common(top_k)]
    return ", ".join(top) if top else "N/A"

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
    target_min = min(max(min_output, 20), max(input_word_count // 3, 15))
    target_max = max(max_output, target_min + 35)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=target_max,
            min_new_tokens=target_min,
            length_penalty=1.0,
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
        combined_text = " ".join(chunk_summaries)
        global_overview = generate_single_summary(
            combined_text,
            lang=resolved_lang,
            max_output=max_output,
            min_output=min_output,
            num_beams=num_beams
        )

        bullets = []
        for ch in chapters:
            t_str = ch.get("time_str", "")
            if ":" in str(t_str):
                time_badge = f"[{t_str}] "
                label = f"Chương {ch['chunk_id']}"
            else:
                time_badge = ""
                label = f"Phần {ch['chunk_id']}"

            bullets.append(f"• **{label} {time_badge}:** {ch['summary']}")

        bullets_joined = "\n".join(bullets)

        if resolved_lang == "vi":
            executive_summary = (
                f"{global_overview}\n\n"
                f"**Tóm lược các phân đoạn chính:**\n"
                f"{bullets_joined}"
            )
        else:
            executive_summary = (
                f"{global_overview}\n\n"
                f"**Key Section Takeaways:**\n"
                f"{bullets_joined}"
            )
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

    report = f"""EVALUATION REPORT (Unicode NFC Preserved)
--------------------------------------------------
Compression Metrics:
  Source Tokens    : {n_t:,}
  Summary Tokens   : {n_s:,}
  Compression Ratio: {ratio * 100:.1f}%

ROUGE Scores (vs. Source):
  Model ROUGE-1    : {r_vs_trans['rouge1'].fmeasure * 100:.1f}%
  Model ROUGE-L    : {r_vs_trans['rougeL'].fmeasure * 100:.1f}%
  Lead-3 ROUGE-1   : {r_lead['rouge1'].fmeasure * 100:.1f}% (Baseline)
  Lead-3 ROUGE-L   : {r_lead['rougeL'].fmeasure * 100:.1f}% (Baseline)

Abstractiveness:
  Novel Vocabulary : {abstractive_ratio:.1f}% (Generated beyond Lead-3)
--------------------------------------------------
Architecture: fine-tuned mT5 + Hierarchical Map-Reduce
Engine: Multilingual Seq2Seq (vi · en · fr · es)"""

    return report
