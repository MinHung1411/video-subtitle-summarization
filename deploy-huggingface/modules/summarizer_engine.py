"""
Summarization Engine Module:
Multilingual Seq2Seq generation using fine-tuned mT5 (minhunhooo/video-subtitle-summarization-mt5)
with Hierarchical Map-Reduce chunking and Unicode-aware ROUGE evaluation.
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
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer as _rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def get_current_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = get_current_device()

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

    curr_device = get_current_device()
    print(f"Loading summarization model: {target_repo}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(target_repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            target_repo,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        _LOADED_MODELS[target_repo] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {target_repo}: {e}. Fallback to csebuetnlp/mT5_multilingual_XLSum...")
        fallback = "csebuetnlp/mT5_multilingual_XLSum"
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback, torch_dtype=torch.float32)
        if torch.cuda.is_available():
            model = model.to("cuda")
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
    max_output: int = 150,
    min_output: int = 30,
    num_beams: int = 4
) -> str:
    model, tokenizer = get_summarizer_model()
    device = get_current_device()
    try:
        model = model.to(device)
    except Exception:
        pass

    clean_text = re.sub(r"\s+", " ", text).strip()
    prefix = f"summarize {lang}: "
    prompt = prefix + clean_text

    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_output,
            min_length=min_output,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def hierarchical_summarize(
    chunks: List[Dict],
    target_lang: str = "auto",
    detected_lang: str = "en",
    max_output: int = 250,
    min_output: int = 50,
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

    for chunk in chunks:
        c_text = chunk["text"]
        time_str = chunk.get("time_str", "")

        c_summary = generate_single_summary(
            c_text,
            lang=resolved_lang,
            max_output=max(80, max_output // 2),
            min_output=min(20, min_output // 2),
            num_beams=num_beams
        )

        chunk_summaries.append(c_summary)
        chapters.append({
            "chunk_id": chunk.get("chunk_id", len(chapters) + 1),
            "time_str": time_str,
            "summary": c_summary,
            "raw_text": c_text
        })

    if len(chunk_summaries) > 1:
        combined_text = " ".join(chunk_summaries)
        executive_summary = generate_single_summary(
            combined_text,
            lang=resolved_lang,
            max_output=max_output,
            min_output=min_output,
            num_beams=num_beams
        )
    else:
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
Engine: fine-tuned mT5 (minhunhooo/video-subtitle-summarization-mt5)"""

    return report
