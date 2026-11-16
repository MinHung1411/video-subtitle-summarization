---
title: Video Subtitle Multilingual Summarization
emoji: 🎥
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
---

# 🎥 Video Subtitle Intelligence & Multilingual Summarization

**Production NLP Architecture · Topic 17 · Industrial University of Ho Chi Minh City (IUH)**

An end-to-end multi-stage pipeline designed for long-form video transcript processing, multilingual speech-to-text, hierarchical map-reduce chunking, and abstractive summarization.

---

## 🌟 Live Demos & Artifacts
- 🚀 **Direct Web Application:** [https://minhunhooo-video-subtitle-summarization.hf.space/](https://minhunhooo-video-subtitle-summarization.hf.space/)
- 🧠 **Pre-trained & Fine-tuned Model Hub:** [minhunhooo/video-subtitle-summarization-mt5](https://huggingface.co/minhunhooo/video-subtitle-summarization-mt5) (Weights, SafeTensors, Benchmark Tables & Visualizations)
- 🐙 **GitHub Repository:** [MinHung1411/video-subtitle-summarization](https://github.com/MinHung1411/video-subtitle-summarization)

---

## 🚀 Key Architectural Highlights

1. **Ingestion Engine (`yt-dlp` & Local I/O):**
   - High-throughput audio stream extraction from YouTube, TikTok, and Vimeo.
   - Support for 4 distinct input modalities: direct video/audio uploads (`.mp4`, `.mkv`, `.mp3`, `.wav`), video URLs, subtitle files (`.srt`, `.vtt`), and raw plain text.

2. **ASR with Temporal Alignment (OpenAI Whisper):**
   - Timestamp-aware acoustic transcription.
   - Extracts sentence boundaries and exact timestamps for timeline visualization.
   - Automatic bracket noise suppression (`[âm nhạc]`, `♪♫`) and YouTube promotional outro stripping.
   - Context-aware phonetic speech rectification for spoken Vietnamese.

3. **Hierarchical Map-Reduce Chunker:**
   - Overcomes standard Transformer context length limits ($512$ tokens).
   - Segments $30-60+$ minute transcripts into cohesive sections, summarizes each independently ("Map"), then synthesizes the global executive summary ("Reduce").

4. **Fine-Tuned Multilingual Seq2Seq Backbone (`mT5`):**
   - Trained across 44,000 multi-domain samples (VT-SSum, VietNews, CNN/DailyMail, OrangeSum, MLSUM).
   - Conditioned via language control prefix: `summarize <lang>: <text>` for Vietnamese, English, French, and Spanish.
   - Post-generation Vietnamese polish engine eliminating awkward literal translations and preserving natural phrasing.

---

## 📊 Empirical Benchmarks

| Metric / Dataset | Zero-shot mT5 | Old BART-large Baseline | **Our Fine-Tuned mT5** | Gain (vs BART) |
| :--- | :---: | :---: | :---: | :---: |
| **VT-SSum (ROUGE-L)** | 11.55% | 24.98% | **33.65%** | **+8.67%** |
| **Macro ROUGE-1** | 29.41% | - | **33.19%** | **+3.78%** |
| **Macro ROUGE-2** | 13.82% | - | **16.03%** | **+2.21%** |
| **Macro ROUGE-L** | 22.65% | - | **26.99%** | **+4.34%** |
| **Faithfulness Score** | - | - | **98.2% – 100%** | State-of-the-art |

---

## 🛠️ Space Architecture

```
.
├── app.py                     # Gradio multi-tab reactive interface
├── requirements.txt           # Python dependencies
├── apt.txt                    # System dependencies (ffmpeg)
├── README.md                  # Space metadata card
└── modules/                   # Production modular pipeline
    ├── audio_handler.py       # Audio stream ingestion via yt-dlp
    ├── asr_engine.py          # Whisper transcription & timestamp parser
    ├── chunker.py             # Time-aligned segment chunking
    ├── summarizer_engine.py   # mT5 generation & map-reduce logic
    └── exporter.py            # Formatted Markdown report export
```

*Note: Model weights are loaded dynamically from Hugging Face Hub (`minhunhooo/video-subtitle-summarization-mt5`) at runtime to keep repository storage lightweight and fast to clone.*
