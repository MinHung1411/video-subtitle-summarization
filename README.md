<div align="center">

# 🎥 Video Subtitle Understanding & Summarization

**NLP Project – Topic 17 · IUH 2026**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

*An end-to-end NLP pipeline that transcribes video audio with Whisper ASR and generates abstractive summaries using a fine-tuned BART-large-cnn model.*

</div>

---

## 📌 Overview

This project builds a complete **video subtitle summarization system** capable of:

- 🎙️ **Transcribing** video/audio content using OpenAI Whisper (ASR)
- ✂️ **Cleaning** raw SRT/VTT subtitles by removing timestamps and noise
- 📝 **Summarizing** transcripts with a fine-tuned `BART-large-cnn` model
- 📊 **Evaluating** output quality using ROUGE, METEOR, and BERTScore
- 🖥️ **Serving** an interactive Gradio UI with three input modes

The model was fine-tuned on the **VT-SSum** dataset (6,000 video transcript–summary pairs) and achieves competitive benchmark results on its 400-sample test set.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                        │
│   📹 Video File  │  📁 SRT/TXT File  │  📝 Raw Text    │
└────────┬────────────────┬─────────────────┬─────────────┘
         │                │                 │
         ▼                ▼                 │
  ┌─────────────┐  ┌─────────────┐         │
  │  Whisper    │  │  SRT Parser │         │
  │  ASR Model  │  │  & Cleaner  │         │
  └──────┬──────┘  └──────┬──────┘         │
         │                │                 │
         └────────────────┴────────────────-┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Text Preprocessing  │
              │  (clean_text, nltk)   │
              └──────────┬────────────┘
                         │
              ┌──────────┴────────────┐
              │                       │
              ▼                       ▼
    ┌──────────────────┐   ┌──────────────────┐
    │  BART-large-cnn  │   │  Lead-3 Baseline │
    │  (Fine-tuned on  │   │  (Extractive)    │
    │    VT-SSum)      │   └──────────────────┘
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Evaluation      │
    │  ROUGE / METEOR  │
    │  / BERTScore     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   Gradio UI      │
    │  (3-tab demo)    │
    └──────────────────┘
```

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Multi-input Support** | Raw text, `.srt`/`.txt` files, or direct video upload |
| **Whisper ASR** | Auto speech recognition with `tiny` / `base` / `small` model options |
| **Abstractive Summarization** | BART fine-tuned for video transcript domain |
| **Lead-3 Baseline** | Extractive baseline for comparison |
| **Keyword Extraction** | TF-based keyword extraction (stopword-filtered) |
| **ROUGE Evaluation** | Real-time ROUGE-1, ROUGE-2, ROUGE-L scoring |
| **Compression Ratio** | Tracks compression ratio vs. VT-SSum benchmark (~11%) |
| **Gradio Interface** | Interactive 3-tab web UI (Text / File / Video) |

---

## 📊 Model Performance

Fine-tuned `BART-large-cnn` evaluated on **400-sample VT-SSum test set**:

| Metric | Score |
|---|---|
| **ROUGE-1** | 42.16% |
| **ROUGE-2** | 21.62% |
| **ROUGE-L** | 24.98% |
| **METEOR** | 26.19% |
| **BERTScore F1** | 0.8555 |

> The model achieves a compression ratio of ~11%, consistent with VT-SSum benchmark expectations.

---

## 🗂️ Project Structure

```
video-subtitle-summarization/
│
├── DEMO_CODE.ipynb        # Main notebook: UI + full inference pipeline
├── CODE_MODEL.ipynb       # Model training & fine-tuning notebook
├── requirements.txt       # Python dependencies
├── Dataset.txt            # Link to VT-SSum dataset on Kaggle
└── README.md
```

---

## ⚙️ Setup & Usage

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- `ffmpeg` installed system-wide

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

On Linux/Colab, also install ffmpeg:

```bash
apt-get install -y ffmpeg
```

### 2. Download the fine-tuned BART model

The fine-tuned model weights (`bart_best.zip`) are stored on Google Drive (IUH NLP shared folder). Place the unzipped `bart_best/` directory at `/content/bart_best` (Colab) or update `BART_MODEL_PATH` in the notebook.

> **Note:** Model weights are not included in this repository due to file size constraints (~800MB). Contact the authors for access.

### 3. Run the Demo

Open `DEMO_CODE.ipynb` in **Google Colab** (GPU runtime recommended) and run all cells in order:

| Cell | Description |
|---|---|
| **Cell 1** | Install libraries + detect CUDA device |
| **Cell 2** | Mount Google Drive + extract BART model |
| **Cell 3** | Load fine-tuned BART model |
| **Cell 4** | Define pipeline functions (clean, summarize, evaluate) |
| **Cell 5** | Launch Gradio web UI |

The Gradio interface will be available at a public `gradio.live` URL (shared link valid for 72h).

---

## 🖥️ Gradio Interface

The demo provides **3 input tabs**:

### 📝 Tab 1 — Raw Transcript Input
Paste any transcript text directly. Supports adjustable parameters:
- Max / Min output tokens
- Number of beams (beam search)

### 📁 Tab 2 — Upload SRT / TXT File
Upload subtitle files (`.srt`, `.txt`, `.vtt`). The pipeline automatically strips timestamps and noise before summarizing.

### 🎬 Tab 3 — Upload Video (Whisper ASR)
Upload a video file (`.mp4`, `.mkv`, `.avi`). Whisper transcribes the audio, then BART summarizes the transcript.

---

## 🧠 Model Details

### Fine-tuned Model: `BART-large-cnn`

| Property | Value |
|---|---|
| **Base model** | `facebook/bart-large-cnn` |
| **Parameters** | ~406M |
| **Training dataset** | VT-SSum (6,000 pairs) |
| **Max input length** | 1,024 tokens |
| **Decoding** | Beam search, `num_beams=4`, `length_penalty=2.0` |
| **No-repeat n-gram** | 3 |
| **Repetition penalty** | 1.2 |

### Whisper ASR

| Property | Value |
|---|---|
| **Library** | `openai-whisper` |
| **Available sizes** | `tiny`, `base`, `small` |
| **Language detection** | Automatic |
| **Inference mode** | `fp32`, `beam_size=1` for speed |

---

## 📦 Dataset

**VT-SSum** — Video Transcript Summarization dataset

- 📎 [Kaggle Link](https://www.kaggle.com/datasets/prasadmagdum/nlpproject)
- ~6,000 video transcript–summary pairs
- Domain: YouTube educational videos
- Language: English

---

## 🔬 Technical Notes

- **Abstractive vs. Extractive**: BART generates novel sentences rather than copying from the transcript. The evaluation measures the percentage of words in BART output *not* present in Lead-3 baseline (abstractive ratio).
- **Chunking**: Input is truncated to 1,024 tokens to match training configuration. For very long videos, consider chunked summarization.
- **GPU Memory**: BART-large with `float16` requires ~3GB VRAM; CPU fallback uses `float32` (~12GB RAM).

---

## 👥 Team

**IUH – Faculty of Information Technology | NLP Course 2026**

> Instructor: **TS. Bùi Thanh Hùng**

| Member | Responsibilities | Hours | % |
|---|---|---|---|
| **Hồ Minh Hưng** | Data EDA, VT-SSum preprocessing, dataset loading pipeline | 40h | 25% |
| **Hồ Minh Hưng** | Fine-tuning BART-large-cnn, Trainer API config, training curves | 60h | 37.5% |
| **Huỳnh Quốc Anh** | T5-small & Lead-3 baseline implementation, metrics computation & comparison | 40h | 25% |
| **Huỳnh Quốc Anh** | Technical report writing, research paper, system diagrams & flowcharts | 20h | 12.5% |
| **Total** | — | **160h** | **100%** |

---

## 📄 License

This project is for academic and portfolio purposes. Model weights derived from `facebook/bart-large-cnn` are subject to the [MIT License](https://github.com/facebookresearch/fairseq/blob/main/LICENSE).

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ at IUH · 2026

</div>
