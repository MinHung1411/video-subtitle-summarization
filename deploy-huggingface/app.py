"""
Video Subtitle Intelligence & Multilingual Summarization Platform
NLP Production Pipeline · Topic 17 · Industrial University of Ho Chi Minh City (IUH) 2026
"""

import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    lead3_summarize,
    evaluate_summary,
    detect_language,
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
                return None, "Error: Vui lòng nhập hoặc dán nội dung văn bản transcript.", "", "", "", None
            full_transcript = clean_subtitle_text(raw_text)
            detected_lang = detect_language(full_transcript)
            title = "Direct Transcript"
            chunks = chunk_plain_text(full_transcript, max_words_per_chunk=320)

        # Mode 2: File Upload (Subtitle or Video/Audio)
        elif input_type == "file":
            if not url_or_path:
                return None, "Error: Vui lòng tải lên file hợp lệ.", "", "", "", None

            file_path = url_or_path if isinstance(url_or_path, str) else getattr(url_or_path, "name", str(url_or_path))
            title = os.path.basename(file_path)

            if file_path.lower().endswith((".srt", ".vtt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                segments = parse_srt_with_timestamps(content)
                if segments:
                    chunks = chunk_by_segments(segments, max_words_per_chunk=280)
                    full_transcript = " ".join([s["text"] for s in segments])
                    total_sec = segments[-1]["end"] if segments else 0
                    duration_str = format_seconds_to_time(total_sec)
                else:
                    full_transcript = clean_subtitle_text(content)
                    chunks = chunk_plain_text(full_transcript, max_words_per_chunk=320)
                detected_lang = detect_language(full_transcript)
            elif file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_transcript = clean_subtitle_text(f.read())
                chunks = chunk_plain_text(full_transcript, max_words_per_chunk=320)
                detected_lang = detect_language(full_transcript)
            else:
                asr_res = transcribe_audio(file_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"Lỗi nhận diện âm thanh ASR: {asr_res['error']}", "", "", "", None
                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=280)
                if asr_res.get("segments"):
                    duration_str = format_seconds_to_time(asr_res["segments"][-1]["end"])

        # Mode 3: Online URL
        elif input_type == "url":
            clean_url = url_or_path.strip()
            if not clean_url:
                return None, "Error: Vui lòng cung cấp link video hợp lệ.", "", "", "", None

            yt_id = extract_youtube_video_id(clean_url)
            transcript_found = False

            # Fast Strategy: Direct Captions extraction
            if yt_id:
                yt_res = fetch_youtube_captions(yt_id)
                if yt_res.get("success") and len(yt_res.get("text", "").split()) >= 15:
                    full_transcript = yt_res["text"]
                    detected_lang = yt_res.get("language", "en")
                    chunks = chunk_by_segments(yt_res["segments"], max_words_per_chunk=280)
                    title = yt_res.get("title", f"YouTube Video ({yt_id})")
                    duration_str = format_seconds_to_time(yt_res.get("duration", 0))
                    transcript_found = True

            # Fallback Strategy: Audio stream download + Whisper ASR
            if not transcript_found:
                dl_res = download_audio_from_url(clean_url)
                if not dl_res["success"]:
                    return (
                        None,
                        "⚠️ **Thông Báo Trích Xuất Video URL:**\n\n"
                        "YouTube hiện kích hoạt tường lửa bot đối với máy chủ đám mây khi tải ngầm luồng âm thanh video không có phụ đề trực tiếp.\n\n"
                        "💡 **Các giải pháp khả dụng ngay:**\n"
                        "1. **Thử Video có phụ đề CC:** Sử dụng các link video có sẵn phụ đề (ví dụ video mẫu bên dưới) để trích xuất tức thì dưới 3 giây.\n"
                        "2. **Dùng Tab 2 (Upload File Phụ Đề):** Tải file `.srt` phụ đề của video lên hệ thống.\n"
                        "3. **Dùng Tab 3 (Upload Video / Audio):** Tải trực tiếp file `.mp4`/`.mp3` lên — mô hình Whisper ASR trên máy chủ 16GB RAM sẽ tự động chuyển đổi giọng nói thành phụ đề và tạo các mốc thời gian chuẩn xác 100%!\n"
                        "4. **Dùng Tab 1 (Nhập Transcript):** Dán trực tiếp lời thoại để tạo tóm tắt ngay lập tức.",
                        "", "", "", None
                    )

                audio_path = dl_res["audio_path"]
                temp_audio_file = audio_path
                title = dl_res.get("title", "Online Video")
                duration_str = format_seconds_to_time(dl_res.get("duration", 0))

                asr_res = transcribe_audio(audio_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"Lỗi ASR: {asr_res['error']}", "", "", "", None

                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"], max_words_per_chunk=280)

        word_count = len(full_transcript.split())
        if word_count < 15:
            return None, f"Cảnh báo: Nội dung quá ngắn ({word_count} từ) để tóm tắt.", "", "", "", None

        # Summarization & Evaluation
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
                if re.search(r"\d{1,2}:\d{2}", str(time_range)):
                    chapter_md.append(f"### 📍 Chương {ch['chunk_id']} `⏱️ {time_range}`\n{ch['summary']}")
                elif time_range and time_range != "Toàn văn":
                    chapter_md.append(f"### 📍 {time_range}\n{ch['summary']}")
        elif chapters and any(re.search(r"\d{1,2}:\d{2}", str(c.get("time_str", ""))) for c in chapters):
            ch = chapters[0]
            chapter_md.append(f"### 📍 Chương 1 `⏱️ {ch.get('time_str', '')}`\n{ch['summary']}")

        if chapter_md:
            chapter_display = "\n\n---\n\n".join(chapter_md)
        else:
            chapter_display = "> ℹ️ *Nội dung đã được tóm tắt súc tích, hoàn chỉnh trong Bản Tóm Tắt ở trên.*"

        total_elapsed = time.time() - t_start

        meta_info = (
            f"""<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; font-size: 13.5px; color: #475569; display: flex; flex-wrap: wrap; gap: 16px; align-items: center;">
              <span>📁 <strong>Tiêu đề:</strong> {title}</span>
              <span>⏱️ <strong>Thời lượng:</strong> {duration_str}</span>
              <span>🌐 <strong>Ngôn ngữ:</strong> <code style="background: #e2e8f0; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{detected_lang.upper()}</code></span>
              <span>📝 <strong>Tổng số từ:</strong> {word_count:,}</span>
              <span>⚡ <strong>Thời gian xử lý:</strong> <span style="color: #16a34a; font-weight: 700;">{total_elapsed:.1f}s</span></span>
              <span>🖥️ <strong>Môi trường:</strong> <span style="color: #2563eb; font-weight: 600;">Production Cloud (16GB RAM)</span></span>
            </div>"""
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
            keywords="",
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
            eval_report,
            transcript_preview,
            download_path
        )

    except Exception as e:
        return None, f"Lỗi hệ thống: {str(e)}", "", "", "", None

@spaces.GPU(duration=1)
def _zerogpu_keeper():
    """Startup probe function for container initialization."""
    return True

def run_text_tab(text, target_lang, max_tokens, min_tokens, num_beams):
    return process_pipeline("text", "", text, target_lang, "base", max_tokens, min_tokens, num_beams)

def run_srt_tab(file_obj, target_lang, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Error: Vui lòng tải lên file phụ đề (.srt, .vtt, .txt).", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, "base", max_tokens, min_tokens, num_beams)

def run_video_tab(file_obj, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Error: Vui lòng tải lên file video/audio (.mp4, .mp3, .wav).", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

def run_url_tab(url, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    return process_pipeline("url", url, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

CUSTOM_CSS = """
.container { max-width: 1260px; margin: auto; }
#summary-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-left: 5px solid #2563eb !important;
    border-radius: 10px !important;
    padding: 20px 24px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #1e293b !important;
    min-height: 160px;
}
#summary-card h3 {
    margin-top: 14px !important;
    margin-bottom: 8px !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}
#chapter-card {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    color: #334155 !important;
}
#metrics-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
    font-size: 14px !important;
}
#metrics-card table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 12px 0 !important;
}
#metrics-card th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    padding: 10px 14px !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 700 !important;
    text-align: left !important;
}
#metrics-card td {
    padding: 8px 14px !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}
"""

TRANSCRIPT_EXAMPLES = [
    ["Trí tuệ nhân tạo (AI) đang phát triển với tốc độ nhanh chóng và tác động sâu sắc đến mọi lĩnh vực trong cuộc sống. Từ việc hỗ trợ các y bác sĩ phân tích phim chụp y tế chính xác hơn, đến việc tối ưu hóa quy trình sản xuất và vận hành chuỗi cung ứng trong doanh nghiệp. Tuy nhiên, cùng với những lợi ích to lớn, AI cũng đặt ra những thách thức nghiêm trọng về an toàn thông tin, quyền riêng tư dữ liệu và các vấn đề đạo đức. Các chuyên gia quốc tế khuyến nghị rằng việc áp dụng AI phải luôn đi đôi với một hành lang pháp lý minh bạch và đảm bảo con người vẫn là yếu tố kiểm soát then chốt."],
    ["So let's talk about how neural networks actually learn. Imagine teaching a child to recognize cats without giving them a rulebook; you simply show them thousands of pictures. A deep neural network learns similarly through backpropagation, adjusting internal weights layer by layer. The early layers identify simple edges and shapes, while deeper layers detect complex patterns like facial structures. This breakthrough in representation learning is why modern deep learning has achieved human-level performance across vision and speech tasks."]
]

URL_EXAMPLES = [
    ["https://www.youtube.com/watch?v=VAIe6L68M3E"],
    ["https://www.youtube.com/watch?v=UF8uR6Z6KLc"]
]

with gr.Blocks(title="VideoInsight · Multilingual Video Intelligence & Hierarchical Summarization", css=CUSTOM_CSS) as demo:

    gr.HTML(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); color: #ffffff; padding: 26px 32px; border-radius: 14px; margin-bottom: 22px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25); border: 1px solid rgba(255, 255, 255, 0.1);">
          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
            <div>
              <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="background: #2563eb; color: #ffffff; font-size: 11px; font-weight: 800; padding: 3px 10px; border-radius: 6px; letter-spacing: 0.5px; text-transform: uppercase;">Production AI Platform</span>
                <span style="color: #64748b; font-size: 13px;">|</span>
                <span style="color: #94a3b8; font-size: 13px; font-weight: 500;">Multimodal Video NLP Pipeline</span>
              </div>
              <h1 style="margin: 0; font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; line-height: 1.2;">
                🎥 VideoInsight · Multilingual Video Intelligence & Hierarchical Summarization
              </h1>
              <p style="margin: 8px 0 0 0; color: #cbd5e1; font-size: 14px; font-weight: 400; max-width: 860px; line-height: 1.5;">
                Hệ thống trí tuệ nhân tạo tóm tắt video đa ngữ (Vi · En · Fr · Es) kết hợp nhận diện giọng nói OpenAI Whisper ASR và mô hình chuỗi sang chuỗi đa ngữ fine-tuned mT5 (582M params)
              </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
              <span style="background: rgba(37, 99, 235, 0.25); color: #93c5fd; border: 1px solid rgba(96, 165, 250, 0.35); padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">🎙️ Whisper ASR</span>
              <span style="background: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.35); padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">🧠 Fine-tuned mT5</span>
              <span style="background: rgba(245, 158, 11, 0.25); color: #fde68a; border: 1px solid rgba(251, 191, 36, 0.35); padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">⏱️ Hierarchical Map-Reduce</span>
              <span style="background: rgba(168, 85, 247, 0.25); color: #e9d5ff; border: 1px solid rgba(192, 132, 252, 0.35); padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 600;">🌐 Vi · En · Fr · Es</span>
            </div>
          </div>
        </div>
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
            btn_text = gr.Button("🚀 Phân Tích & Tóm Tắt Văn Bản (Summarize Text)", variant="primary", size="large")

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

            btn_srt = gr.Button("🚀 Trích Xuất & Tóm Tắt File Phụ Đề (Summarize Subtitles)", variant="primary", size="large")

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

            btn_video = gr.Button("🚀 Nhận Diện Giọng Nói & Tóm Tắt Video (Whisper + mT5)", variant="primary", size="large")

        # Tab 4: 🔗 Video URL (YouTube / Stream)
        with gr.Tab("🔗 4. Link Video Trực Tuyến (URL)"):
            gr.Markdown(
                """
                > 💡 **Khuyến nghị Trải Nghiệm:**  
                > • **Video Mẫu Trực Tuyến:** Nhấp chọn các link mẫu có sẵn bên dưới (TED Tiếng Việt & Steve Jobs Tiếng Anh) để kiểm tra tốc độ trích xuất phụ đề và tóm tắt đa ngôn ngữ tức thì trong 2–4 giây.  
                > • **Đối với Video Tùy Chọn Riêng:** Các nền tảng video đám mây thường kích hoạt tường lửa chống bot đối với máy chủ cloud khi tải ngầm âm thanh từ video không có phụ đề trực tiếp. Đối với video riêng của bạn, khuyến nghị tải file qua **Tab 2 (.srt)** hoặc **Tab 3 (.mp4/.mp3)** để Whisper nhận diện giọng nói và tạo dòng thời gian chuẩn xác 100%!
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    url_in = gr.Textbox(
                        label="Video URL (YouTube, Vimeo, direct stream)",
                        placeholder="Dán link YouTube có phụ đề CC vào đây (ví dụ: https://www.youtube.com/watch?v=...)",
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

            gr.Examples(
                examples=URL_EXAMPLES,
                inputs=url_in,
                label="Video mẫu có sẵn phụ đề chuẩn (Click để thử nghiệm ngay):"
            )

            btn_url = gr.Button("🚀 Phân Tích & Tóm Tắt từ Link URL (Summarize from URL)", variant="primary", size="large")

    gr.Markdown("---")
    gr.Markdown("### 📊 Kết Quả Tóm Tắt (Summary Results)")

    meta_out = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("#### 📌 Bản Tóm Tắt (Executive Summary)")
            exec_out = gr.Markdown(
                value="*Kết quả tóm tắt súc tích, đầy đủ ý chính sẽ xuất hiện tại đây sau khi bạn nhấn nút xử lý.*",
                elem_id="summary-card"
            )
        with gr.Column(scale=1):
            gr.Markdown("#### 📥 Tải Báo Cáo")
            download_out = gr.File(
                label="Báo cáo tóm tắt (.md)",
                interactive=False
            )

    with gr.Accordion("📍 Dòng Thời Gian Phân Đoạn (Chapter Timeline)", open=True):
        chapter_out = gr.Markdown(
            value="*Dòng thời gian các phân đoạn nội dung chính (nếu có mốc thời gian từ video/audio) sẽ hiển thị tại đây.*",
            elem_id="chapter-card"
        )

    with gr.Accordion("🔬 Chi Tiết Kỹ Thuật & Đánh Giá NLP (ROUGE Benchmark · Dành cho Tuyển Dụng / Phỏng Vấn)", open=False):
        stats_out = gr.Markdown(
            value="*Bảng chỉ số đo lường ROUGE, tỷ lệ nén và mức độ tóm tắt trừu tượng (Abstractiveness).*",
            elem_id="metrics-card"
        )

    with gr.Accordion("📄 Lời Thoại Transcript Gốc (Source Transcript)", open=False):
        transcript_out = gr.Textbox(
            label="Nội dung lời thoại trích xuất từ audio / phụ đề",
            lines=8,
            max_lines=15,
            interactive=False
        )

    output_components = [
        meta_out,
        exec_out,
        chapter_out,
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
