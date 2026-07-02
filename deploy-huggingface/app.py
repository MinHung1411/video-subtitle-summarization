"""
Video Subtitle Understanding & Summarization
NLP Project – Topic 17 · IUH 2026
Model: BART-large-cnn fine-tuned on VT-SSum

Deploy version for Hugging Face Spaces (CPU Free tier).
"""

import os
import re
import time
from collections import Counter

import gradio as gr
import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer as _rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ====================== SETUP ======================
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Thư mục chứa model đã fine-tune, đặt cùng cấp với app.py trong repo Space
BART_MODEL_PATH = os.environ.get("BART_MODEL_PATH", "bart_best")

print(f"Loading BART model from: {BART_MODEL_PATH}")
bart_tokenizer = AutoTokenizer.from_pretrained(BART_MODEL_PATH)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(
    BART_MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
bart_model.eval()

n_params = sum(p.numel() for p in bart_model.parameters()) / 1e6
print(f"BART model loaded. Params: {n_params:.0f}M | Device: {DEVICE}")

_ROUGE = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "the", "a", "an", "is", "it", "in", "of", "and", "to", "for",
        "with", "was", "are", "be", "this", "that",
    }


# ====================== TEXT CLEANING ======================
def clean_text(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ====================== KEYWORD EXTRACTION ======================
def extract_keywords(text, top_k=12):
    words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in STOP_WORDS and len(w) > 3]
    freq = Counter(words)
    top = [w for w, _ in freq.most_common(top_k)]
    return ", ".join(top) if top else "(không tìm được từ khóa)"


# ====================== LEAD-3 ======================
def lead3_summarize(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:3]) if sentences else text[:500]


# ====================== BART SUMMARIZE ======================
def bart_summarize(transcript, max_output=256, min_output=60, num_beams=4):
    t0 = time.time()
    inputs = bart_tokenizer(
        transcript, max_length=1024, truncation=True, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = bart_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_output,
            min_length=min_output,
            num_beams=num_beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
            forced_bos_token_id=0,
        )
    summary = bart_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return summary, time.time() - t0


# ====================== EVALUATE ======================
def evaluate_summary(transcript: str, bart_s: str, lead3: str):
    words_t = transcript.lower().split()
    words_s = bart_s.lower().split()
    words_l = lead3.lower().split()
    n_t, n_s = len(words_t), len(words_s)

    ratio = n_s / max(n_t, 1)

    r_vs_trans = _ROUGE.score(transcript[:2000], bart_s)
    r_lead_vs_trans = _ROUGE.score(transcript[:2000], lead3)

    set_lead = set(words_l)
    non_extractive = [w for w in words_s if w not in set_lead and w.isalpha() and len(w) > 2]
    abstractive_ratio = len(non_extractive) / max(n_s, 1) * 100

    bart_sents = [s.strip().lower() for s in bart_s.split(".") if s.strip()]
    lead_sents = [s.strip().lower() for s in lead3.split(".") if s.strip()]
    novel_sents = sum(1 for s in bart_sents if not any(s in l or l in s for l in lead_sents))
    sent_novelty = novel_sents / max(len(bart_sents), 1) * 100

    comment = f"""📊 Phân tích Output
{'─' * 50}
📏 Độ nén
  • Input          : {n_t:,} từ
  • BART output    : {n_s:,} từ
  • Tỉ lệ nén      : {ratio * 100:.1f}%  (benchmark VT-SSum: ~11%)

📐 ROUGE vs Transcript gốc
  • BART  ROUGE-1  : {r_vs_trans['rouge1'].fmeasure * 100:.1f}%
  • BART  ROUGE-L  : {r_vs_trans['rougeL'].fmeasure * 100:.1f}%
  • Lead-3 ROUGE-1 : {r_lead_vs_trans['rouge1'].fmeasure * 100:.1f}%  (baseline extractive)
  • Lead-3 ROUGE-L : {r_lead_vs_trans['rougeL'].fmeasure * 100:.1f}%  (baseline extractive)

🔬 Tính Abstractive (so với Lead-3)
  • Từ khác Lead-3 : {abstractive_ratio:.1f}%  ← BART diễn đạt khác baseline
  • Câu mới hoàn toàn: {sent_novelty:.1f}%  ← không copy nguyên câu Lead-3
{'─' * 50}
ℹ️  Kết quả trên tập test (400 mẫu — VT-SSum):
  ROUGE-1=42.16% | ROUGE-2=21.62% | ROUGE-L=24.98%
  METEOR=26.19%  | BERTScore-F1=0.8555"""

    return comment


# ====================== PIPELINE ======================
def run_pipeline_text(raw_text, max_output=256, min_output=60, num_beams=4):
    if not raw_text or not raw_text.strip():
        return ("", "", "⚠️ Vui lòng nhập transcript!", "", "")
    cleaned = clean_text(raw_text)
    word_count = len(cleaned.split())
    if word_count < 20:
        return (cleaned, "", f"⚠️ Transcript quá ngắn ({word_count} từ).", "", "")

    keywords = extract_keywords(cleaned)
    lead3 = lead3_summarize(cleaned)
    bart_s, t = bart_summarize(cleaned, max_output, min_output, num_beams)
    eval_comment = evaluate_summary(cleaned, bart_s, lead3)

    stats = f"{eval_comment}\n\n⏱️ Thời gian sinh: {t:.2f}s | Device: {DEVICE.upper()}"
    preview = cleaned[:2000] + ("..." if len(cleaned) > 2000 else "")
    return (preview, keywords, bart_s, lead3, stats)


# ====================== WHISPER (lazy-loaded) ======================
WHISPER_MODEL = None
WHISPER_MODEL_SIZE = "base"


def load_whisper(size="base"):
    global WHISPER_MODEL, WHISPER_MODEL_SIZE
    if WHISPER_MODEL is None or size != WHISPER_MODEL_SIZE:
        import whisper

        print(f"Loading Whisper {size}...")
        WHISPER_MODEL = whisper.load_model(size)
        WHISPER_MODEL_SIZE = size
        print(f"Whisper {size} loaded!")
    return WHISPER_MODEL


def run_pipeline_video(video_path, max_output=256, min_output=60, num_beams=4, whisper_size="base"):
    if video_path is None:
        return ("", "", "⚠️ Chưa upload video!", "", "")
    try:
        wm = load_whisper(whisper_size)
        t_start = time.time()
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"Video size: {file_size_mb:.1f} MB")
        result = wm.transcribe(
            video_path,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0,
            beam_size=1,
            best_of=1,
            language=None,
        )
        transcript = result["text"]
        lang = result.get("language", "?")
        asr_time = time.time() - t_start
        print(f"Transcribed {len(transcript.split())} từ ({asr_time:.1f}s)")
        preview, keywords, bart_s, lead3_s, stats = run_pipeline_text(
            transcript, max_output, min_output, num_beams
        )
        stats = f"🎙️ ASR: {asr_time:.1f}s | Ngôn ngữ: {lang} | File: {file_size_mb:.1f}MB\n" + stats
        return (preview, keywords, bart_s, lead3_s, stats)
    except Exception as e:
        return ("", "", f"❌ Lỗi: {str(e)}", "", "")


print("✅ Pipeline loaded — BART fine-tuned on VT-SSum | Abstractive Summarization")


# ====================== GRADIO UI ======================
def gradio_text(text, max_tokens, min_tokens, num_beams):
    return run_pipeline_text(text, max_output=max_tokens, min_output=min_tokens, num_beams=num_beams)


def gradio_srt(file_obj):
    if file_obj is None:
        return ("", "", "⚠️ Chưa upload file!", "", "")
    try:
        raw = open(file_obj.name, encoding="utf-8", errors="ignore").read()
        return gradio_text(raw, 300, 80, 6)
    except Exception as e:
        return ("", "", f"❌ Lỗi: {e}", "", "")


def gradio_video(video_path, max_tokens, min_tokens, num_beams, whisper_size="base"):
    global WHISPER_MODEL, WHISPER_MODEL_SIZE
    if whisper_size != WHISPER_MODEL_SIZE:
        WHISPER_MODEL = None
    WHISPER_MODEL_SIZE = whisper_size
    return run_pipeline_video(video_path, max_tokens, min_tokens, num_beams, whisper_size)


CSS = """
#bart-output textarea { background:#f0fff4 !important; border-left:4px solid #38a169 !important; font-size: 15px; }
#lead3-output textarea { background:#fffaf0 !important; border-left:4px solid #dd6b20 !important; }
#stats-output textarea { background:#ebf8ff !important; border-left:4px solid #3182ce !important; }
"""

# Transcript mẫu (văn phong nói / bài giảng, gần với domain VT-SSum) để người dùng
# bấm thử nhanh thay vì phải tự tìm text dán vào.
TRANSCRIPT_EXAMPLES = [
    ["""So let's talk about how neural networks actually learn. Imagine you're teaching a child to recognize cats. You don't give them a rulebook, you just show them hundreds of pictures and say this is a cat, this is not a cat. A neural network learns pretty much the same way. It starts out completely random, guessing wildly, and every time it gets an answer wrong, it adjusts its internal weights just a tiny bit to get closer to the right answer next time. This process is called backpropagation, and it happens millions of times during training. Now, the interesting part is that nobody actually programs the rules for what makes a cat look like a cat. The network figures that out on its own, layer by layer. The first layers might learn to detect edges and simple shapes, the middle layers combine those into patterns like fur texture or eye shapes, and the deeper layers start recognizing whole objects. This is why we call it deep learning, because of all those stacked layers. Of course, this only works if you have enough data and enough compute power, which is exactly why deep learning only really took off in the last decade or so, once GPUs became powerful and cheap enough."""],
    ["""Now here's something a lot of people don't realize about solar panels. The efficiency of a typical solar panel today is only around twenty percent, which means eighty percent of the sunlight hitting it is basically wasted as heat. But that number has actually doubled in the last fifteen years, and researchers are pushing new materials like perovskite that could push efficiency even higher while making panels cheaper to produce. So why does this matter? Well, the cost of solar electricity has dropped by almost ninety percent since two thousand and nine, and in many parts of the world it's now the cheapest source of new electricity generation, even cheaper than coal or gas. That's a massive shift in just over a decade. Of course, solar isn't perfect. The sun doesn't shine at night, and that's where battery storage comes in. As battery prices keep falling too, we're getting closer to a grid that can run reliably on renewables alone. The transition is happening faster than most predictions from just five or six years ago, which is honestly one of the more hopeful stories in the whole climate conversation."""],
    ["""Let's talk about why sleep is so much more complicated than just closing your eyes for eight hours. Your brain actually cycles through several stages every night, roughly ninety minutes at a time. There's light sleep, deep sleep, and REM sleep, and each one does something completely different. Deep sleep is when your body repairs muscle tissue and your brain clears out waste products that build up during the day. REM sleep, on the other hand, is when most of your dreaming happens, and it seems to play a huge role in consolidating memories and processing emotions. Here's the tricky part though. Early in the night you get more deep sleep, but as the night goes on, you get more REM sleep. So if you cut your night short, say you only sleep five hours instead of eight, you're not just losing three hours evenly, you're disproportionately losing REM sleep, which is exactly the part your brain needs for emotional regulation and memory. That's part of why people who chronically undersleep often report feeling foggy, irritable, or forgetful, even if they don't feel physically exhausted."""],
]

with gr.Blocks(title="🎥 Video Summarization - Topic 17", theme=gr.themes.Soft(primary_hue="purple"), css=CSS) as demo:

    gr.Markdown(
        """
    # 🎥 Video Subtitle Understanding & Summarization
    **NLP Project – Topic 17 · IUH 2026**
    Mô hình: `BART-large-cnn` fine-tuned trên VT-SSum

    ⚠️ Space đang chạy trên **CPU Free** nên xử lý video/Whisper có thể mất vài phút.
    """
    )

    with gr.Tabs():

        # Tab 1: Nhập Transcript
        with gr.Tab("📝 Nhập Transcript"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt_in = gr.Textbox(
                        label="📄 Transcript đầu vào",
                        lines=12,
                        placeholder="Dán transcript hoặc phụ đề vào đây...",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**⚙️ Tham số**")
                    sl_max_t = gr.Slider(60, 256, value=200, step=20, label="Max output tokens")
                    sl_min_t = gr.Slider(30, 100, value=60, step=10, label="Min output tokens")
                    sl_beam_t = gr.Slider(2, 8, value=4, step=1, label="Num beams")

            gr.Examples(
                examples=TRANSCRIPT_EXAMPLES,
                inputs=txt_in,
                label="💡 Ví dụ nhanh (click để điền vào ô transcript, sau đó bấm Tóm tắt)",
            )

            btn_txt = gr.Button("🚀 Tóm tắt", variant="primary", size="large")

            with gr.Row():
                out_preview_t = gr.Textbox(label="📜 Transcript (preview)", lines=5, interactive=False)
                out_kw_t = gr.Textbox(label="🔑 Từ khóa chính", lines=3, interactive=False)

            with gr.Row():
                out_bart_t = gr.Textbox(label="📌 BART Summary (Đề xuất)", lines=8, interactive=False, elem_id="bart-output")
                out_lead3_t = gr.Textbox(label="📋 Lead-3 Baseline", lines=6, interactive=False, elem_id="lead3-output")

            out_stats_t = gr.Textbox(label="📊 Thống kê & Đánh giá", lines=12, interactive=False, elem_id="stats-output")

            btn_txt.click(
                fn=gradio_text,
                inputs=[txt_in, sl_max_t, sl_min_t, sl_beam_t],
                outputs=[out_preview_t, out_kw_t, out_bart_t, out_lead3_t, out_stats_t],
            )

        # Tab 2: Upload File
        with gr.Tab("📁 Upload File (.srt / .txt)"):
            file_in = gr.File(label="Upload file phụ đề hoặc transcript", file_types=[".srt", ".txt", ".vtt"])
            btn_file = gr.Button("🚀 Xử lý & Tóm tắt", variant="primary")

            with gr.Row():
                out_preview_f = gr.Textbox(label="📜 Transcript (preview)", lines=5, interactive=False)
                out_kw_f = gr.Textbox(label="🔑 Từ khóa chính", lines=3, interactive=False)
            with gr.Row():
                out_bart_f = gr.Textbox(label="📌 BART Summary", lines=8, interactive=False, elem_id="bart-output")
                out_lead3_f = gr.Textbox(label="📋 Lead-3 Baseline", lines=6, interactive=False, elem_id="lead3-output")
            out_stats_f = gr.Textbox(label="📊 Thống kê & Đánh giá", lines=12, interactive=False, elem_id="stats-output")

            btn_file.click(
                fn=gradio_srt,
                inputs=[file_in],
                outputs=[out_preview_f, out_kw_f, out_bart_f, out_lead3_f, out_stats_f],
            )

        # Tab 3: Upload Video
        with gr.Tab("🎬 Upload Video (Whisper ASR)"):
            with gr.Row():
                vid_in = gr.Video(label="📹 Upload Video (mp4, mkv, avi)")
                with gr.Column():
                    gr.Markdown("**⚙️ Tham số**")
                    sl_max_v = gr.Slider(60, 256, value=200, step=20, label="Max output tokens")
                    sl_min_v = gr.Slider(30, 100, value=60, step=10, label="Min output tokens")
                    sl_beam_v = gr.Slider(2, 8, value=4, step=1, label="Num beams")
                    dd_whisper = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="tiny",
                        label="Whisper Model (tiny = nhanh nhất, khuyến nghị trên CPU)",
                    )

            btn_vid = gr.Button("🔊 Transcribe → Tóm tắt", variant="primary", size="large")

            with gr.Row():
                out_preview_v = gr.Textbox(label="📜 Transcript (Whisper)", lines=6, interactive=False)
                out_kw_v = gr.Textbox(label="🔑 Từ khóa chính", lines=3, interactive=False)
            with gr.Row():
                out_bart_v = gr.Textbox(label="📌 BART Summary", lines=8, interactive=False, elem_id="bart-output")
                out_lead3_v = gr.Textbox(label="📋 Lead-3 Baseline", lines=6, interactive=False, elem_id="lead3-output")
            out_stats_v = gr.Textbox(label="📊 Thống kê & Đánh giá", lines=12, interactive=False, elem_id="stats-output")

            btn_vid.click(
                fn=gradio_video,
                inputs=[vid_in, sl_max_v, sl_min_v, sl_beam_v, dd_whisper],
                outputs=[out_preview_v, out_kw_v, out_bart_v, out_lead3_v, out_stats_v],
            )

    gr.Markdown(
        "---\n**Mô hình**: BART-large-cnn fine-tuned trên VT-SSum (6.000 mẫu) | "
        "ROUGE-1=42.16% | ROUGE-L=24.98% | BERTScore=0.8555"
    )

if __name__ == "__main__":
    demo.launch()
