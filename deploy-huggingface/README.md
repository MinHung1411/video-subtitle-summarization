---
title: Video Subtitle Summarization
emoji: 🎥
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# 🎥 Video Subtitle Understanding & Summarization

NLP Project – Topic 17 · IUH 2026
Mô hình: `BART-large-cnn` fine-tuned trên VT-SSum (ROUGE-1=42.16%, ROUGE-L=24.98%, BERTScore=0.8555)

## Cấu trúc thư mục cần có trong Space

```
.
├── app.py
├── requirements.txt
├── apt.txt
└── bart_best/            <-- copy thư mục model đã fine-tune vào đây
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ... (các file tokenizer khác)
```
