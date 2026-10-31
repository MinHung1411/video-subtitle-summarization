"""
Video Subtitle Understanding & Multilingual Summarization Suite
NLP Production Pipeline · Topic 17 · IUH 2026
"""

import os
import time
import gradio as gr
from modules.audio_handler import download_audio_from_url, format_seconds_to_time
from modules.asr_engine import transcribe_audio, clean_subtitle_text, parse_srt_with_timestamps
from modules.chunker import chunk_by_segments, chunk_plain_text
from modules.summarizer_engine import (
    hierarchical_summarize,
    extract_keywords,
    lead3_summarize,
    evaluate_summary,
    DEVICE
)
from modules.exporter import format_markdown_report, create_downloadable_file

def process_pipeline(
    input_type: str,
    url_or_path: str,
    raw_text: str,
    target_lang: str = "Auto",
    whisper_size: str = "base",
    max_tokens: int = 250,
    min_tokens: int = 60,
    num_beams: int = 4
):
    t_start = time.time()
    title = "Analysis Session"
    duration_str = "N/A"
    detected_lang = "en"
    temp_audio_file = None
    chunks = []
    full_transcript = ""

    lang_map = {
        "Auto": "auto",
        "Tiếng Việt": "vi",
        "English": "en",
        "Français": "fr",
        "Español": "es",
        "Tieng Viet": "vi"
    }
    target_lang_code = lang_map.get(target_lang, "auto")

    try:
        # Ingestion
        if input_type == "url":
            if not url_or_path or not url_or_path.strip():
                return None, "Error: Please provide a valid video URL.", "", "", "", "", None

            dl_res = download_audio_from_url(url_or_path.strip())
            if not dl_res["success"]:
                return None, f"Audio extraction error: {dl_res['error']}", "", "", "", "", None

            audio_path = dl_res["audio_path"]
            temp_audio_file = audio_path
            title = dl_res.get("title", "YouTube Video")
            duration_str = format_seconds_to_time(dl_res.get("duration", 0))

            asr_res = transcribe_audio(audio_path, model_size=whisper_size)
            if not asr_res["success"]:
                return None, f"ASR transcription error: {asr_res['error']}", "", "", "", "", None

            full_transcript = asr_res["text"]
            detected_lang = asr_res["language"]
            chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=400)

        elif input_type == "file":
            if not url_or_path:
                return None, "Error: Please upload a video, audio, or subtitle file.", "", "", "", "", None

            file_path = url_or_path
            title = os.path.basename(file_path)

            if file_path.endswith((".srt", ".vtt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                segments = parse_srt_with_timestamps(content)
                if segments:
                    chunks = chunk_by_segments(segments, max_words_per_chunk=400)
                    full_transcript = " ".join([s["text"] for s in segments])
                else:
                    full_transcript = clean_subtitle_text(content)
                    chunks = chunk_plain_text(full_transcript, max_words_per_chunk=400)
            else:
                asr_res = transcribe_audio(file_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"ASR transcription error: {asr_res['error']}", "", "", "", "", None
                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=400)

        elif input_type == "text":
            if not raw_text or not raw_text.strip():
                return None, "Error: Input text is empty.", "", "", "", "", None
            full_transcript = clean_subtitle_text(raw_text)
            title = "Text Transcript"
            chunks = chunk_plain_text(full_transcript, max_words_per_chunk=400)

        word_count = len(full_transcript.split())
        if word_count < 15:
            return None, f"Warning: Transcript is too short ({word_count} words) for summarization.", "", "", "", "", None

        # Summarization & Evaluation
        keywords = extract_keywords(full_transcript)
        lead3 = lead3_summarize(full_transcript)

        exec_summary, chapters, sum_time = hierarchical_summarize(
            chunks=chunks,
            target_lang=target_lang_code,
            detected_lang=detected_lang,
            max_output=max_tokens,
            min_output=min_tokens,
            num_beams=num_beams
        )

        eval_report = evaluate_summary(full_transcript, exec_summary, lead3)

        # Chapter Timeline Formatting
        chapter_md = []
        if chapters and len(chapters) > 1:
            for ch in chapters:
                time_range = ch.get("time_str", "")
                chapter_md.append(f"- **`{time_range}` Chapter {ch['chunk_id']}:** {ch['summary']}")
        elif chapters:
            chapter_md.append(f"- **Summary:** {chapters[0]['summary']}")
        chapter_display = "\n\n".join(chapter_md)

        total_elapsed = time.time() - t_start

        meta_info = (
            f"**Title:** {title} | **Duration:** {duration_str} | **Language:** {detected_lang.upper()} | "
            f"**Total Words:** {word_count:,} | **Latency:** {total_elapsed:.1f}s | **Device:** {DEVICE.upper()}"
        )

        metadata_dict = {
            "duration_str": duration_str,
            "language": detected_lang,
            "word_count": word_count
        }
        full_md_report = format_markdown_report(
            title=title,
            executive_summary=exec_summary,
            chapters=chapters,
            keywords=keywords,
            metadata=metadata_dict
        )
        download_path = create_downloadable_file(full_md_report, filename_prefix="Summary_Report")

        # Cleanup temporary audio
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except Exception:
                pass

        transcript_preview = full_transcript[:1500] + ("..." if len(full_transcript) > 1500 else "")

        return (
            meta_info,
            exec_summary,
            chapter_display,
            keywords,
            eval_report,
            transcript_preview,
            download_path
        )

    except Exception as e:
        return None, f"System error: {str(e)}", "", "", "", "", None

def run_url_tab(url, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    return process_pipeline("url", url, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

def run_file_tab(file_obj, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Please upload a file.", "", "", "", "", None
    return process_pipeline("file", file_obj.name, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

def run_text_tab(text, target_lang, max_tokens, min_tokens, num_beams):
    return process_pipeline("text", "", text, target_lang, "base", max_tokens, min_tokens, num_beams)

CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
#summary-box textarea { background: #f8fafc !important; border-left: 4px solid #2563eb !important; font-size: 14.5px; line-height: 1.6; }
#chapter-box { background: #ffffff; border-radius: 6px; padding: 12px; border: 1px solid #e2e8f0; }
#stats-box textarea { font-family: monospace; background: #f8fafc !important; font-size: 12.5px; }
"""

TRANSCRIPT_EXAMPLES = [
    ["Trí tuệ nhân tạo (AI) đang phát triển với tốc độ nhanh chóng và tác động sâu sắc đến mọi lĩnh vực trong cuộc sống. Từ việc hỗ trợ các y bác sĩ phân tích phim chụp y tế chính xác hơn, đến việc tối ưu hóa quy trình sản xuất và vận hành chuỗi cung ứng trong doanh nghiệp. Tuy nhiên, cùng với những lợi ích to lớn, AI cũng đặt ra những thách thức nghiêm trọng về an toàn thông tin, quyền riêng tư dữ liệu và các vấn đề đạo đức. Các chuyên gia quốc tế khuyến nghị rằng việc áp dụng AI phải luôn đi đôi với một hành lang pháp lý minh bạch và đảm bảo con người vẫn là yếu tố kiểm soát then chốt."],
    ["So let's talk about how neural networks actually learn. Imagine teaching a child to recognize cats without giving them a rulebook; you simply show them thousands of pictures. A deep neural network learns similarly through backpropagation, adjusting internal weights layer by layer. The early layers identify simple edges and shapes, while deeper layers detect complex patterns like facial structures. This breakthrough in representation learning is why modern deep learning has achieved human-level performance across vision and speech tasks."]
]

with gr.Blocks(title="Video Multilingual Summarization Suite", theme=gr.themes.Base(), css=CUSTOM_CSS) as demo:

    gr.Markdown(
        """
        # 🎥 Video Subtitle Understanding & Multilingual Summarization
        **NLP Production Pipeline · Topic 17 · IUH 2026**
        
        End-to-end framework integrating speech-to-text (Whisper), hierarchical chunking, and multilingual abstractive summarization.
        
        🔗 **Model Hub:** [`minhunhooo/video-subtitle-summarization-mt5`](https://huggingface.co/minhunhooo/video-subtitle-summarization-mt5) | 💻 **GitHub:** [`video-subtitle-summarization`](https://github.com/MinHung1411/video-subtitle-summarization)
        """
    )

    with gr.Tabs():

        # Tab 1: YouTube / Web URL
        with gr.Tab("YouTube / Web URL"):
            with gr.Row():
                with gr.Column(scale=3):
                    url_in = gr.Textbox(
                        label="Video URL (YouTube, TikTok, Vimeo)",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    lang_opt_1 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Output Summary Language"
                    )
                    whisper_opt_1 = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="base",
                        label="Whisper Model"
                    )

            with gr.Accordion("Generation Hyperparameters", open=False):
                with gr.Row():
                    sl_max_1 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_1 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_1 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            btn_url = gr.Button("Summarize Video", variant="primary", size="large")

        # Tab 2: Upload File
        with gr.Tab("Upload File"):
            with gr.Row():
                with gr.Column(scale=3):
                    file_in = gr.File(
                        label="Upload Video, Audio, or Subtitle file (.mp4, .mkv, .mp3, .wav, .srt)",
                        file_types=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a", ".srt", ".vtt"]
                    )
                with gr.Column(scale=1):
                    lang_opt_2 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Output Summary Language"
                    )
                    whisper_opt_2 = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="base",
                        label="Whisper Model"
                    )

            with gr.Accordion("Generation Hyperparameters", open=False):
                with gr.Row():
                    sl_max_2 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_2 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_2 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            btn_file = gr.Button("Summarize File", variant="primary", size="large")

        # Tab 3: Text Transcript
        with gr.Tab("Direct Transcript"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_in = gr.Textbox(
                        label="Transcript / Raw Subtitle Text",
                        placeholder="Paste lecture transcript or subtitle text here...",
                        lines=10
                    )
                with gr.Column(scale=1):
                    lang_opt_3 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Output Summary Language"
                    )
                    sl_max_3 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_3 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_3 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            gr.Examples(examples=TRANSCRIPT_EXAMPLES, inputs=text_in, label="Examples:")
            btn_text = gr.Button("Summarize Text", variant="primary", size="large")

    gr.Markdown("---")
    gr.Markdown("### Summary & Analysis Results")

    meta_out = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=2):
            exec_out = gr.Textbox(
                label="Executive Summary",
                lines=7,
                interactive=False,
                elem_id="summary-box"
            )
        with gr.Column(scale=1):
            kw_out = gr.Textbox(
                label="Key Concepts",
                lines=3,
                interactive=False
            )
            download_out = gr.File(
                label="Export Markdown Report (.md)",
                interactive=False
            )

    with gr.Accordion("Chapter Timeline", open=True):
        chapter_out = gr.Markdown(elem_id="chapter-box")

    with gr.Row():
        with gr.Column():
            stats_out = gr.Textbox(
                label="Evaluation Metrics (ROUGE & Compression)",
                lines=10,
                interactive=False,
                elem_id="stats-box"
            )
        with gr.Column():
            transcript_out = gr.Textbox(
                label="Source Transcript (Preview)",
                lines=10,
                interactive=False
            )

    output_components = [
        meta_out,
        exec_out,
        chapter_out,
        kw_out,
        stats_out,
        transcript_out,
        download_out
    ]

    btn_url.click(
        fn=run_url_tab,
        inputs=[url_in, lang_opt_1, whisper_opt_1, sl_max_1, sl_min_1, sl_beam_1],
        outputs=output_components
    )

    btn_file.click(
        fn=run_file_tab,
        inputs=[file_in, lang_opt_2, whisper_opt_2, sl_max_2, sl_min_2, sl_beam_2],
        outputs=output_components
    )

    btn_text.click(
        fn=run_text_tab,
        inputs=[text_in, lang_opt_3, sl_max_3, sl_min_3, sl_beam_3],
        outputs=output_components
    )

if __name__ == "__main__":
    demo.launch()
