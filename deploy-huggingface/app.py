"""
Video Subtitle Understanding & Multilingual Summarization Suite
NLP Production Pipeline · Topic 17 · Industrial University of Ho Chi Minh City (IUH) 2026
"""

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is not None and callable(fn):
                return fn
            def decorator(func):
                return func
            return decorator

import os
import time
import gradio as gr
from modules.audio_handler import (
    download_audio_from_url,
    format_seconds_to_time,
    extract_youtube_video_id,
    fetch_youtube_captions
)
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
    title = "Session Analysis"
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
        # Mode 1: Text Transcript
        if input_type == "text":
            if not raw_text or not raw_text.strip():
                return None, "Error: Vui lòng nhập hoặc dán nội dung văn bản transcript.", "", "", "", "", None
            full_transcript = clean_subtitle_text(raw_text)
            title = "Direct Transcript"
            chunks = chunk_plain_text(full_transcript, max_words_per_chunk=400)

        # Mode 2: File Upload (Subtitle or Video/Audio)
        elif input_type == "file":
            if not url_or_path:
                return None, "Error: Vui lòng tải lên file hợp lệ.", "", "", "", "", None

            file_path = url_or_path if isinstance(url_or_path, str) else getattr(url_or_path, "name", str(url_or_path))
            title = os.path.basename(file_path)

            if file_path.lower().endswith((".srt", ".vtt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                segments = parse_srt_with_timestamps(content)
                if segments:
                    chunks = chunk_by_segments(segments, max_words_per_chunk=400)
                    full_transcript = " ".join([s["text"] for s in segments])
                    total_sec = segments[-1]["end"] if segments else 0
                    duration_str = format_seconds_to_time(total_sec)
                else:
                    full_transcript = clean_subtitle_text(content)
                    chunks = chunk_plain_text(full_transcript, max_words_per_chunk=400)
            elif file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_transcript = clean_subtitle_text(f.read())
                chunks = chunk_plain_text(full_transcript, max_words_per_chunk=400)
            else:
                asr_res = transcribe_audio(file_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"Lỗi nhận diện âm thanh ASR: {asr_res['error']}", "", "", "", "", None
                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=400)
                if asr_res.get("segments"):
                    duration_str = format_seconds_to_time(asr_res["segments"][-1]["end"])

        # Mode 3: Online URL
        elif input_type == "url":
            clean_url = url_or_path.strip()
            if not clean_url:
                return None, "Error: Vui lòng cung cấp link video hợp lệ.", "", "", "", "", None

            yt_id = extract_youtube_video_id(clean_url)
            transcript_found = False

            # Fast Strategy: Direct Captions extraction
            if yt_id:
                yt_res = fetch_youtube_captions(yt_id)
                if yt_res.get("success") and len(yt_res.get("text", "").split()) >= 15:
                    full_transcript = yt_res["text"]
                    detected_lang = yt_res.get("language", "en")
                    chunks = chunk_by_segments(yt_res["segments"], max_words_per_chunk=400)
                    title = yt_res.get("title", f"YouTube Video ({yt_id})")
                    duration_str = format_seconds_to_time(yt_res.get("duration", 0))
                    transcript_found = True

            # Fallback Strategy: Audio stream download + Whisper ASR
            if not transcript_found:
                dl_res = download_audio_from_url(clean_url)
                if not dl_res["success"]:
                    err_msg = dl_res.get("error", "Unknown error")
                    if "Sign in to confirm you're not a bot" in err_msg or "bot" in err_msg.lower():
                        return (
                            None,
                            "⚠️ **Tường lửa YouTube (Datacenter Protection):** YouTube chặn tải ngầm video trực tiếp từ máy chủ đám mây.\n\n"
                            "💡 **Để tiếp tục tóm tắt video này:**\n"
                            "1. Dùng **Tab 2 ('Upload Phụ Đề')**: Tải file `.srt` phụ đề của video lên (rất nhẹ, xử lý ngay).\n"
                            "2. Dùng **Tab 3 ('Upload Video / Audio')**: Tải file `.mp4` hoặc `.mp3` của video lên để Whisper nhận diện.\n"
                            "3. Dùng **Tab 1 ('Nhập Transcript')**: Dán lời thoại vào để mô hình tóm tắt tức thì!",
                            "", "", "", "", None
                        )
                    return None, f"Lỗi trích xuất audio: {err_msg}", "", "", "", "", None

                audio_path = dl_res["audio_path"]
                temp_audio_file = audio_path
                title = dl_res.get("title", "Online Video")
                duration_str = format_seconds_to_time(dl_res.get("duration", 0))

                asr_res = transcribe_audio(audio_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"Lỗi ASR: {asr_res['error']}", "", "", "", "", None

                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=400)

        word_count = len(full_transcript.split())
        if word_count < 15:
            return None, f"Cảnh báo: Nội dung quá ngắn ({word_count} từ) để tóm tắt.", "", "", "", "", None

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
            f"**Tiêu đề:** {title} | **Thời lượng:** {duration_str} | **Ngôn ngữ:** {detected_lang.upper()} | "
            f"**Tổng số từ:** {word_count:,} | **Độ trễ:** {total_elapsed:.1f}s | **Phần cứng:** {DEVICE.upper()} (ZeroGPU A100)"
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
        return None, f"Lỗi hệ thống: {str(e)}", "", "", "", "", None

@spaces.GPU(duration=30)
def run_text_tab(text, target_lang, max_tokens, min_tokens, num_beams):
    return process_pipeline("text", "", text, target_lang, "base", max_tokens, min_tokens, num_beams)

@spaces.GPU(duration=30)
def run_srt_tab(file_obj, target_lang, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Error: Vui lòng tải lên file phụ đề (.srt, .vtt, .txt).", "", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, "base", max_tokens, min_tokens, num_beams)

@spaces.GPU(duration=60)
def run_video_tab(file_obj, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Error: Vui lòng tải lên file video/audio (.mp4, .mp3, .wav).", "", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

@spaces.GPU(duration=60)
def run_url_tab(url, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    return process_pipeline("url", url, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

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

with gr.Blocks(title="Video Subtitle Multilingual Summarization Suite", theme=gr.themes.Base(), css=CUSTOM_CSS) as demo:

    gr.Markdown(
        """
        # 🎥 Video Subtitle Understanding & Multilingual Summarization
        **NLP Production Pipeline · Topic 17 · Industrial University of Ho Chi Minh City (IUH)**
        
        End-to-end framework integrating speech-to-text (Whisper), hierarchical chunking, and multilingual abstractive summarization (`mT5`).
        
        🔗 **Model Hub:** [`minhunhooo/video-subtitle-summarization-mt5`](https://huggingface.co/minhunhooo/video-subtitle-summarization-mt5) | 💻 **GitHub:** [`video-subtitle-summarization`](https://github.com/MinHung1411/video-subtitle-summarization)
        """
    )

    with gr.Tabs():

        # Tab 1: 📝 Nhập / Dán Transcript (Default & Instant)
        with gr.Tab("📝 1. Nhập / Dán Transcript"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_in = gr.Textbox(
                        label="Transcript / Raw Subtitle Text (Văn bản lời thoại)",
                        placeholder="Dán nội dung phụ đề bài giảng hoặc transcript video vào đây...",
                        lines=10
                    )
                with gr.Column(scale=1):
                    lang_opt_1 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Ngôn ngữ tóm tắt đầu ra"
                    )
                    sl_max_1 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_1 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_1 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            gr.Examples(examples=TRANSCRIPT_EXAMPLES, inputs=text_in, label="Dữ liệu mẫu thử nhanh (Click để chọn):")
            btn_text = gr.Button("Tóm tắt Văn bản (Summarize Text)", variant="primary", size="large")

        # Tab 2: 📁 Upload File Phụ Đề (.srt / .vtt / .txt)
        with gr.Tab("📁 2. Upload File Phụ Đề (.srt / .txt)"):
            with gr.Row():
                with gr.Column(scale=3):
                    srt_in = gr.File(
                        label="Tải lên file phụ đề (.srt, .vtt, .txt)",
                        file_types=[".srt", ".vtt", ".txt"],
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    lang_opt_2 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Ngôn ngữ tóm tắt đầu ra"
                    )
                    sl_max_2 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_2 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_2 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            btn_srt = gr.Button("Tóm tắt File Phụ Đề (Summarize Subtitles)", variant="primary", size="large")

        # Tab 3: 🎬 Upload Video / Audio (Whisper ASR)
        with gr.Tab("🎬 3. Upload Video / Audio (Whisper ASR)"):
            with gr.Row():
                with gr.Column(scale=3):
                    video_in = gr.File(
                        label="Tải lên file Video hoặc Âm thanh (.mp4, .mkv, .avi, .mp3, .wav, .m4a)",
                        file_types=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a"],
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    lang_opt_3 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Ngôn ngữ tóm tắt đầu ra"
                    )
                    whisper_opt_3 = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="base",
                        label="Mô hình Whisper ASR"
                    )

            with gr.Accordion("Siêu tham số sinh (Generation Hyperparameters)", open=False):
                with gr.Row():
                    sl_max_3 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_3 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_3 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            btn_video = gr.Button("Nhận diện giọng nói & Tóm tắt Video", variant="primary", size="large")

        # Tab 4: 🔗 Video URL (YouTube / Stream)
        with gr.Tab("🔗 4. Link Video Trực Tuyến (URL)"):
            with gr.Row():
                with gr.Column(scale=3):
                    url_in = gr.Textbox(
                        label="Video URL (YouTube, Vimeo, TikTok, direct stream)",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    lang_opt_4 = gr.Dropdown(
                        choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                        value="Auto",
                        label="Ngôn ngữ tóm tắt đầu ra"
                    )
                    whisper_opt_4 = gr.Dropdown(
                        choices=["tiny", "base", "small"],
                        value="base",
                        label="Mô hình Whisper ASR"
                    )

            with gr.Accordion("Siêu tham số sinh (Generation Hyperparameters)", open=False):
                with gr.Row():
                    sl_max_4 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                    sl_min_4 = gr.Slider(30, 150, value=60, step=10, label="Min Tokens")
                    sl_beam_4 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

            btn_url = gr.Button("Tóm tắt từ Link URL", variant="primary", size="large")

    gr.Markdown("---")
    gr.Markdown("### 📊 Kết Quả Tóm Tắt & Phân Tích (Summary & Analysis Results)")

    meta_out = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=2):
            exec_out = gr.Textbox(
                label="Executive Summary (Bản tóm tắt điều hành)",
                lines=7,
                interactive=False,
                elem_id="summary-box"
            )
        with gr.Column(scale=1):
            kw_out = gr.Textbox(
                label="Key Concepts (Từ khóa chính)",
                lines=3,
                interactive=False
            )
            download_out = gr.File(
                label="Tải báo cáo tóm tắt (.md)",
                interactive=False
            )

    with gr.Accordion("Dòng thời gian phân đoạn (Chapter Timeline)", open=True):
        chapter_out = gr.Markdown(elem_id="chapter-box")

    with gr.Row():
        with gr.Column():
            stats_out = gr.Textbox(
                label="Báo cáo đánh giá (ROUGE & Tỷ lệ nén)",
                lines=10,
                interactive=False,
                elem_id="stats-box"
            )
        with gr.Column():
            transcript_out = gr.Textbox(
                label="Văn bản gốc (Source Transcript Preview)",
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

    btn_text.click(
        fn=run_text_tab,
        inputs=[text_in, lang_opt_1, sl_max_1, sl_min_1, sl_beam_1],
        outputs=output_components,
        api_name="summarize_text"
    )

    btn_srt.click(
        fn=run_srt_tab,
        inputs=[srt_in, lang_opt_2, sl_max_2, sl_min_2, sl_beam_2],
        outputs=output_components,
        api_name="summarize_srt"
    )

    btn_video.click(
        fn=run_video_tab,
        inputs=[video_in, lang_opt_3, whisper_opt_3, sl_max_3, sl_min_3, sl_beam_3],
        outputs=output_components,
        api_name="summarize_video"
    )

    btn_url.click(
        fn=run_url_tab,
        inputs=[url_in, lang_opt_4, whisper_opt_4, sl_max_4, sl_min_4, sl_beam_4],
        outputs=output_components,
        api_name="summarize_url"
    )

if __name__ == "__main__":
    demo.queue().launch()
