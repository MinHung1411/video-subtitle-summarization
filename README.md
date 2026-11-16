<div align="center">

# 🎥 Video Subtitle Understanding & Multilingual Summarization

**Production NLP Pipeline · Topic 17 · Industrial University of Ho Chi Minh City (IUH)**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-mT5--Multilingual-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/minhunhooo/video-subtitle-summarization-mt5)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/minhunhooo/video-subtitle-summarization)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/MinHung1411/video-subtitle-summarization)

*An end-to-end multi-stage NLP platform that ingests video streams directly via URL or local files, transcribes timestamped speech with OpenAI Whisper, chunks long contexts with Hierarchical Map-Reduce, and generates abstractive summaries in multiple languages (Vietnamese, English, French, Spanish).*

</div>

---

## 🌟 Live Demos & Artifacts

- 🚀 **Interactive Hugging Face Space:** [minhunhooo/video-subtitle-summarization](https://huggingface.co/spaces/minhunhooo/video-subtitle-summarization)
- 🧠 **Pre-trained & Fine-tuned Model Hub:** [minhunhooo/video-subtitle-summarization-mt5](https://huggingface.co/minhunhooo/video-subtitle-summarization-mt5) (Weights, SafeTensors, Benchmark Tables & Visualizations)
- 📊 **Training Pipeline:** Kaggle Dual-GPU T4 distributed multi-session training with Async Hub Sync.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          INGESTION LAYER                               │
│  📹 Web / YouTube URL (yt-dlp)  │  📁 Local Media File (.mp4, .mp3)   │
│  📁 Subtitles (.srt, .vtt)       │  📝 Plain Text Transcript           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ASR & TEMPORAL SEGMENTATION                        │
│                   OpenAI Whisper (tiny / base / small)                 │
│         Outputs time-aligned segments: {start, end, text}              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL MAP-REDUCE CHUNKER                      │
│   Splits transcripts into semantic 400-token chunks with timestamps    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               MULTILINGUAL ABSTRACTIVE SUMMARIZER (mT5)                │
│     Prompt Routing: "summarize <vi|en|fr|es>: <transcript_chunk>"      │
│     - MAP PHASE   : Summarize each segment -> Chapter Timeline         │
│     - REDUCE PHASE: Synthesize segment summaries -> Executive Summary   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION & PRESENTATION                       │
│  - Unicode-aware NFC ROUGE (preserves Vietnamese accents)              │
│  - Key concepts & extractive novelty ratio                             │
│  - Interactive Gradio Web UI + Exportable Markdown Report (.md)        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks & Ablation

### 1. In-Domain Video Transcripts (VT-SSum Test Set)

| Model Architecture | Params | ROUGE-1 | ROUGE-2 | ROUGE-L | Gain (vs Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Lead-3 (Extractive Baseline) | - | 19.42% | 8.15% | 15.20% | Baseline |
| Zero-shot mT5-XLSum | 582M | 14.80% | 5.21% | 11.55% | - |
| BART-large-cnn (Previous Baseline) | 406M | 42.16% | 21.62% | 24.98% | +9.78% |
| **Our Fine-Tuned mT5 (Epoch 4)** | **582M** | **38.92%** | **22.45%** | **33.65%** | **+8.67% (vs BART)** |

### 2. Multi-Domain & Multilingual Generalization (Macro Average across 5 Datasets)

Evaluated across **VT-SSum** (Video), **VietNews** (Vietnamese News), **CNN/DailyMail** (English News), **OrangeSum** (French), and **MLSUM** (Spanish):

| Evaluation Strategy | Macro ROUGE-1 | Macro ROUGE-2 | Macro ROUGE-L | Multilingual BERTScore-F1 |
| :--- | :---: | :---: | :---: | :---: |
| Zero-Shot mT5-XLSum | 29.41% | 13.82% | 22.65% | 0.702 |
| Lead-3 Extractive | 24.12% | 10.35% | 18.17% | - |
| **Our Fine-Tuned mT5** | **33.19%** | **16.03%** | **26.99%** | **0.758 (Avg across Vi/En/Fr/Es)** |

### 3. Hierarchical Map-Reduce Ablation (Full-Length Long Content)

| Method | Video Transcripts (VT-SSum) | Long Articles (CNN/DM) |
| :--- | :---: | :---: |
| Truncation (First 512 tokens) | 14.03% | 17.04% |
| **Hierarchical Map-Reduce (Ours)** | **16.37% (+2.34%)** | **21.09% (+4.05%)** |

### 4. Faithfulness & Numeric Hallucination Check
- **VT-SSum:** 100.0% numeric consistency (0 hallucinated numbers).
- **VietNews:** 98.2% numeric consistency.

---

## 🚀 Recent Production Updates & Engineering Highlights

| Feature / Upgrade | Technical Implementation | Impact / Value |
| :--- | :--- | :--- |
| **ASR Noise & Outro Purge** | Regex pattern matching in `asr_engine.py` targeting bracket noise (`[âm nhạc]`, `♪♫`) and YouTube subscription outro ads | Prevents junk promotional text from polluting summaries |
| **Phonetic Speech Rectification** | Contextual phoneme dictionary in `asr_engine.py` for Vietnamese acoustic misrecognitions | Restores correct proper names, technical terms, and speaker titles |
| **Vietnamese Polish Engine** | Grammar & tone refinery in `summarizer_engine.py` (`polish_vietnamese_summary`) | Eliminates stiff or offensive literal phrasing, ensures natural human-level readability |
| **Cohesive Executive Briefing** | Redesigned generation prompt in `app.py` replacing AI-tagged bullet lists with smooth narrative paragraphs | Executive-ready briefing suitable for reports and business summaries |
| **4-Language Presets** | One-click instant sample loaders for Vietnamese, English, French, and Spanish | Allows instant zero-configuration testing for reviewers and recruiters |
| **Interactive ROUGE Benchmark** | Real-time Lead-3 vs fine-tuned mT5 comparison using NFC Unicode-aware tokenization | Demonstrates empirical superiority over extractive baselines on the fly |
| **Full Modular Decoupling** | Modular architecture (`asr_engine`, `chunker`, `summarizer_engine`, `exporter`) | Production-ready, maintainable, easily deployable as microservices |

---

## 🗂️ Project Structure

```
video-subtitle-summarization/
├── deploy-huggingface/             # Production deployment suite for HF Spaces
│   ├── app.py                     # Multi-tab Gradio 5 application
│   ├── requirements.txt           # Production dependencies
│   ├── apt.txt                    # System dependencies (ffmpeg)
│   ├── README.md                  # Spaces metadata card
│   └── modules/                   # Decoupled production modules
│       ├── audio_handler.py       # Audio stream ingestion via yt-dlp
│       ├── asr_engine.py          # Whisper transcription & timestamp parser
│       ├── chunker.py             # Time-aligned segment chunking
│       ├── summarizer_engine.py   # mT5 generation & map-reduce logic
│       └── exporter.py            # Formatted Markdown report export
├── TRAIN_MULTILINGUAL_MT5.ipynb    # 44,000-sample distributed Kaggle training notebook (vi, en, fr, es)
├── CODE_MODEL.ipynb                # Baseline single-language training notebook (BART & T5 on VT-SSum)
├── DEMO_CODE.ipynb                 # Inference & evaluation demonstration notebook
├── deploy_to_space.py             # Automated one-click deployment script
├── CV_PORTFOLIO_GUIDE.md          # Technical interview guide & CV bullet points
├── requirements.txt               # Workspace development dependencies
└── README.md                      # Project documentation & benchmark overview
```

---

## 📂 Datasets & Corpora

- **VT-SSum (Video Transcripts):** [Kaggle Dataset](https://www.kaggle.com/datasets/prasadmagdum/nlpproject) | [GitHub](https://github.com/Dod-o/VT-SSum)
- **VietNews (Vietnamese Abstractive Summarization):** [GitHub](https://github.com/Thanh-Bao-Nguyen/VietNews)
- **XL-Sum (Multilingual Abstractive Summarization):** [Hugging Face Datasets](https://huggingface.co/datasets/csebuetnlp/xlsum)
- **OrangeSum (French) & MLSUM (Spanish):** Integrated for zero-shot and cross-lingual validation.

---

## 💻 Quick Start & Deployment

### Run Locally

1. **Clone repository:**
   ```bash
   git clone https://github.com/MinHung1411/video-subtitle-summarization.git
   cd video-subtitle-summarization
   ```

2. **Install dependencies:**
   ```bash
   pip install -r deploy-huggingface/requirements.txt
   ```

3. **Launch the Web UI:**
   ```bash
   cd deploy-huggingface
   python app.py
   ```

### Deploy to Hugging Face Space

Run the automated deployment script with your Hugging Face write token:

```bash
python deploy_to_space.py --token <YOUR_HF_TOKEN>
```

This will automatically push the clean modular pipeline to `minhunhooo/video-subtitle-summarization` and purge legacy monolithic model weights to ensure immediate container initialization.

---

## 👥 Contributors & Academic Context

**Industrial University of Ho Chi Minh City (IUH) — Faculty of Information Technology**  
*NLP Research Project · Topic 17 · 2026*  
Instructor: **TS. Bùi Thanh Hùng**

- **Hồ Minh Hưng** (`@minhunhooo`): Project Lead, Distributed Model Training, MLOps Hub Sync, Gradio UI & Deployment.
- **Huỳnh Quốc Anh**: Baseline evaluation, report writing, system diagrams.

---

<div align="center">
⭐ Star this repository if you find it helpful!
</div>
