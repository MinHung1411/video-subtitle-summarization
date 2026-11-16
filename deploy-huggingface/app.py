import os
import re
import sys
import time
import json
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(duration=1):
            def decorator(func):
                return func
            return decorator
import gradio as gr
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.audio_handler import (
    format_seconds_to_time,
    download_audio_from_url,
    extract_youtube_video_id,
    fetch_youtube_captions
)
from modules.asr_engine import transcribe_audio, clean_subtitle_text, parse_srt_with_timestamps
from modules.chunker import chunk_by_segments, chunk_plain_text
from modules.summarizer_engine import (
    hierarchical_summarize,
    evaluate_summary,
    detect_language,
    lead3_summarize,
    extract_thematic_points
)
from modules.exporter import format_markdown_report, create_downloadable_file

def process_pipeline(
    input_type: str,
    url_or_path: str,
    raw_text: str,
    target_lang: str,
    whisper_size: str,
    max_tokens: int,
    min_tokens: int,
    num_beams: int
):
    """
    Unified multimodal summarization pipeline:
    Supports Raw Text, Subtitle Files (.srt/.txt), Audio/Video files, and Online URLs.
    """
    t_start = time.time()
    temp_audio_file = None

    try:
        full_transcript = ""
        duration_str = "N/A"
        detected_lang = "vi"
        title = "Nội dung"
        chunks = []

        lang_map = {
            "Auto": "auto",
            "Auto (Detect)": "auto",
            "Tự động (Auto)": "auto",
            "Vietnamese": "vi",
            "Tiếng Việt": "vi",
            "English": "en",
            "Tiếng Anh": "en",
            "French": "fr",
            "Français": "fr",
            "Tiếng Pháp": "fr",
            "Spanish": "es",
            "Español": "es",
            "Tiếng Tây Ban Nha": "es",
            "Tieng Viet": "vi"
        }
        target_lang_code = lang_map.get(target_lang, "auto")

        # Mode 1: Text Transcript
        if input_type == "text":
            if not raw_text or not raw_text.strip():
                return None, "Vui lòng nhập hoặc dán nội dung văn bản transcript.", "", "", "", None
            full_transcript = clean_subtitle_text(raw_text)
            detected_lang = detect_language(full_transcript)
            title = "Văn bản trực tiếp"
            chunks = chunk_plain_text(full_transcript)

        # Mode 2: File Upload (Subtitle or Video/Audio)
        elif input_type == "file":
            if not url_or_path:
                return None, "Vui lòng tải lên file hợp lệ.", "", "", "", None

            file_path = url_or_path if isinstance(url_or_path, str) else getattr(url_or_path, "name", str(url_or_path))
            title = os.path.basename(file_path)

            if file_path.lower().endswith((".srt", ".vtt")):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                segments = parse_srt_with_timestamps(content)
                if segments:
                    chunks = chunk_by_segments(segments)
                    full_transcript = " ".join([s["text"] for s in segments])
                    total_sec = segments[-1]["end"] if segments else 0
                    duration_str = format_seconds_to_time(total_sec)
                else:
                    full_transcript = clean_subtitle_text(content)
                    chunks = chunk_plain_text(full_transcript)
                detected_lang = detect_language(full_transcript)
            elif file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    full_transcript = clean_subtitle_text(f.read())
                chunks = chunk_plain_text(full_transcript)
                detected_lang = detect_language(full_transcript)
            else:
                asr_res = transcribe_audio(file_path, model_size=whisper_size)
                if not asr_res["success"]:
                    return None, f"Lỗi nhận diện âm thanh ASR: {asr_res['error']}", "", "", "", None
                full_transcript = asr_res["text"]
                detected_lang = asr_res["language"]
                chunks = chunk_by_segments(asr_res["segments"])
                if asr_res.get("segments"):
                    duration_str = format_seconds_to_time(asr_res["segments"][-1]["end"])

        # Mode 3: Online URL
        elif input_type == "url":
            clean_url = (url_or_path or "").strip()
            if not clean_url:
                return None, "Vui lòng cung cấp link video hợp lệ.", "", "", "", None

            yt_id = extract_youtube_video_id(clean_url)
            transcript_found = False

            # Fast Strategy: Direct Captions extraction
            if yt_id:
                yt_res = fetch_youtube_captions(yt_id)
                if yt_res.get("success") and len(yt_res.get("text", "").split()) >= 15:
                    full_transcript = yt_res["text"]
                    detected_lang = yt_res.get("language", "en")
                    chunks = chunk_by_segments(yt_res["segments"])
                    title = yt_res.get("title", f"YouTube Video ({yt_id})")
                    duration_str = format_seconds_to_time(yt_res.get("duration", 0))
                    transcript_found = True

            # Fallback Strategy: Audio stream download + Whisper ASR
            if not transcript_found:
                dl_res = download_audio_from_url(clean_url)
                if not dl_res["success"]:
                    return (
                        None,
                        "**Thông báo trích xuất Video URL:**\n\n"
                        "YouTube hiện kích hoạt tường lửa đối với máy chủ đám mây khi tải ngầm luồng âm thanh video không có phụ đề trực tiếp.\n\n"
                        "**Các giải pháp khả dụng ngay:**\n"
                        "1. **Thử Video có phụ đề CC:** Sử dụng các link video có sẵn phụ đề (ví dụ video mẫu bên dưới) để trích xuất tức thì dưới 3 giây.\n"
                        "2. **Dùng Tab Tải lên Video / Audio:** Tải trực tiếp file `.mp4`/`.mp3` lên — mô hình Whisper ASR trên máy chủ 16GB RAM sẽ tự động chuyển đổi giọng nói thành văn bản chuẩn xác 100%.\n"
                        "3. **Dùng Tab File Phụ Đề:** Tải file `.srt` phụ đề của video lên hệ thống.\n"
                        "4. **Dùng Tab Văn bản:** Dán trực tiếp lời thoại để tạo tóm tắt ngay lập tức.",
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
                chunks = chunk_by_segments(asr_res["segments"])

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

        # Chapter Timeline & Thematic Formatting (Google Clean Card Style)
        chapter_md = []
        if chapters and len(chapters) > 1:
            for ch in chapters:
                time_range = ch.get("time_str", "")
                if re.search(r"\d{1,2}:\d{2}", str(time_range)):
                    chapter_md.append(
                        f"""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
  <div style="font-size: 12.5px; font-weight: 600; color: #2563eb; margin-bottom: 4px;">Phân đoạn {ch['chunk_id']} · {time_range}</div>
  <div style="font-size: 14px; color: #1e293b; line-height: 1.6;">{ch['summary']}</div>
</div>"""
                    )
                elif time_range and time_range != "Toàn văn":
                    chapter_md.append(
                        f"""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
  <div style="font-size: 12.5px; font-weight: 600; color: #2563eb; margin-bottom: 4px;">{time_range}</div>
  <div style="font-size: 14px; color: #1e293b; line-height: 1.6;">{ch['summary']}</div>
</div>"""
                    )
        elif chapters and any(re.search(r"\d{1,2}:\d{2}", str(c.get("time_str", ""))) for c in chapters):
            ch = chapters[0]
            chapter_md.append(
                f"""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
  <div style="font-size: 12.5px; font-weight: 600; color: #2563eb; margin-bottom: 4px;">Phân đoạn 1 · {ch.get('time_str', '')}</div>
  <div style="font-size: 14px; color: #1e293b; line-height: 1.6;">{ch['summary']}</div>
</div>"""
            )

        if chapter_md:
            chapter_display = "\n".join(chapter_md)
        else:
            thematic_pts = extract_thematic_points(full_transcript, lang=detected_lang)
            if thematic_pts:
                pts_md = []
                for pt_title, content in thematic_pts:
                    pts_md.append(
                        f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
  <div style="font-size: 12.5px; font-weight: 600; color: #1a73e8; margin-bottom: 4px;">{pt_title}</div>
  <div style="font-size: 14px; color: #202124; line-height: 1.6;">{content}</div>
</div>"""
                    )
                chapter_display = "\n".join(pts_md)
            else:
                chapter_display = f"> *Toàn bộ nội dung ({word_count:,} từ) đã được tổng hợp trong Bản tóm tắt điều hành ở trên.*"

        total_elapsed = time.time() - t_start

        # Clean Cohesive Metadata Bar
        meta_info = (
            f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 8px 14px; margin-top: 8px; font-size: 12.5px; color: #3c4043; display: flex; flex-wrap: wrap; gap: 14px; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
              <span>Nguồn: <strong style="color: #202124;">{title}</strong></span>
              <span>Thời lượng: <strong>{duration_str}</strong></span>
              <span>Ngôn ngữ: <strong style="background: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 4px; font-size: 11.5px; border: 1px solid #d2e3fc;">{detected_lang.upper()}</strong></span>
              <span>Số từ: <strong>{word_count:,}</strong></span>
              <span>Thời gian: <strong style="color: #188038;">{total_elapsed:.1f}s</strong></span>
              <span style="color: #5f6368; font-size: 11.5px; background: #f8f9fa; padding: 2px 8px; border-radius: 4px; border: 1px solid #dadce0;">ZeroGPU</span>
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
            metadata=metadata_dict,
            source_transcript=full_transcript
        )
        download_path = create_downloadable_file(full_md_report, filename_prefix="Summary_Report")

        # Cleanup temporary audio
        if temp_audio_file and os.path.exists(temp_audio_file):
            try:
                os.remove(temp_audio_file)
            except Exception:
                pass

        return (
            meta_info,
            exec_summary,
            chapter_display,
            eval_report,
            full_transcript,
            download_path
        )

    except Exception as e:
        return None, f"Lỗi hệ thống: {str(e)}", "", "", "", None

@spaces.GPU(duration=1)
def _zerogpu_keeper():
    """Startup probe function for container initialization."""
    return True

def run_text_tab(text, target_lang, max_tokens, min_tokens, num_beams):
    meta, summary, chaps, eval_rep, trans, dl = process_pipeline(
        "text", "", text, target_lang, "base", max_tokens, min_tokens, num_beams
    )
    return meta, summary, dl

def run_srt_tab(file_obj, target_lang, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Vui lòng tải lên file phụ đề (.srt, .vtt, .txt).", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, "base", max_tokens, min_tokens, num_beams)

def run_video_tab(file_obj, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    if file_obj is None:
        return None, "Vui lòng tải lên file video/audio (.mp4, .mp3, .wav).", "", "", "", None
    path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", str(file_obj))
    return process_pipeline("file", path, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

def run_url_tab(url, target_lang, whisper_size, max_tokens, min_tokens, num_beams):
    return process_pipeline("url", url, "", target_lang, whisper_size, max_tokens, min_tokens, num_beams)

CUSTOM_CSS = """
/* Google Translate / Material 3 Design System */
body, .gradio-container {
    background-color: #f8fafc !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    color: #202124 !important;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 24px 20px !important;
}

/* Tabs Navigation: Google Translate Pill Tabs - Force Overrides */
div.gradio-container .tabs > div:first-child,
div.gradio-container .tab-nav,
div.gradio-container [role="tablist"] {
    border-bottom: none !important;
    border: none !important;
    background: transparent !important;
    gap: 8px !important;
    margin: 0 0 16px 0 !important;
    padding: 0 !important;
    display: flex !important;
    flex-wrap: wrap !important;
}

div.gradio-container .tab-nav button,
div.gradio-container [role="tablist"] button,
div.gradio-container button[role="tab"] {
    border: 1px solid #dadce0 !important;
    border-bottom: 1px solid #dadce0 !important;
    border-radius: 20px !important;
    color: #3c4043 !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 7px 18px !important;
    background: #ffffff !important;
    transition: all 0.15s ease-in-out !important;
    box-shadow: none !important;
    cursor: pointer !important;
    margin: 0 4px 6px 0 !important;
    outline: none !important;
}

div.gradio-container .tab-nav button:hover,
div.gradio-container [role="tablist"] button:hover,
div.gradio-container button[role="tab"]:hover {
    background: #f1f3f4 !important;
    border-color: #dadce0 !important;
    color: #202124 !important;
}

div.gradio-container .tab-nav button.selected,
div.gradio-container [role="tablist"] button.selected,
div.gradio-container button[role="tab"][aria-selected="true"],
div.gradio-container button[role="tab"].selected {
    background: #e8f0fe !important;
    color: #1a73e8 !important;
    border: 1px solid #d2e3fc !important;
    border-bottom: 1px solid #d2e3fc !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

div.gradio-container .tab-nav button::after,
div.gradio-container .tab-nav button::before,
div.gradio-container button[role="tab"]::after,
div.gradio-container button[role="tab"]::before {
    display: none !important;
    content: none !important;
    border: none !important;
    height: 0 !important;
}

/* Primary Action CTA Buttons */
button.primary, .gr-button-primary {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    letter-spacing: -0.01em !important;
    padding: 9px 18px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
    cursor: pointer !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #1765cc !important;
    box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1) !important;
}
button.primary:active, .gr-button-primary:active {
    background: #1557b0 !important;
    transform: scale(0.99) !important;
}

/* 4 Sample Pills Row - Strictly 1 Single Row, Never Wrap */
.sample-row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.sample-row > div,
.sample-row > button {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    width: 25% !important;
    margin: 0 !important;
}
.sample-pill {
    background: #ffffff !important;
    color: #3c4043 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 11.5px !important;
    padding: 6px 2px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    text-align: center !important;
    min-height: 33px !important;
    height: 33px !important;
    width: 100% !important;
    min-width: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.sample-pill:hover {
    background: #f1f3f4 !important;
    border-color: #bdc1c6 !important;
    color: #202124 !important;
}

/* Language Sample Pills (Tab 2) - 2x2 Grid with Full Text Visibility */
.sample-pill-lang {
    background: #ffffff !important;
    color: #3c4043 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    padding: 7px 10px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    text-align: center !important;
    min-height: 35px !important;
    height: auto !important;
    width: 100% !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.35 !important;
}
.sample-pill-lang:hover {
    background: #f1f3f4 !important;
    border-color: #1a73e8 !important;
    color: #1a73e8 !important;
}

/* Download Button Component */
.dl-pill {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 12.5px !important;
    padding: 6px 12px !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
    transition: all 0.15s ease-in-out !important;
    cursor: pointer !important;
    width: 100% !important;
    justify-content: center !important;
    min-height: 33px !important;
    height: 33px !important;
}
.dl-pill:hover {
    background: #e8f0fe !important;
    border-color: #d2e3fc !important;
    color: #1765cc !important;
}

/* Tab 1: Dual Comparison Cards - 100% Equal Height & Pixel-Perfect Alignment */
.source-box.block, .summary-box.block {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    height: 300px !important;
    min-height: 300px !important;
    max-height: 300px !important;
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    display: flex !important;
    flex-direction: column !important;
}

.source-box.block > div,
.source-box.block .wrap,
.source-box.block label {
    height: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.source-box.block textarea {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: #ffffff !important;
    height: 100% !important;
    width: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #202124 !important;
    resize: none !important;
    box-sizing: border-box !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: #bdc1c6 #f8f9fa !important;
}

/* Force visible, clean scrollbars across all browsers */
div.gradio-container .source-box textarea::-webkit-scrollbar,
div.gradio-container .source-box textarea.scroll-hide::-webkit-scrollbar,
div.gradio-container .summary-box::-webkit-scrollbar,
div.gradio-container .summary-box-tab2::-webkit-scrollbar,
div.gradio-container .scroll-hide::-webkit-scrollbar {
    display: block !important;
    width: 8px !important;
    background: #f1f3f4 !important;
}

div.gradio-container .source-box textarea::-webkit-scrollbar-track,
div.gradio-container .source-box textarea.scroll-hide::-webkit-scrollbar-track,
div.gradio-container .summary-box::-webkit-scrollbar-track,
div.gradio-container .summary-box-tab2::-webkit-scrollbar-track,
div.gradio-container .scroll-hide::-webkit-scrollbar-track {
    background: #f1f3f4 !important;
    border-radius: 4px !important;
}

div.gradio-container .source-box textarea::-webkit-scrollbar-thumb,
div.gradio-container .source-box textarea.scroll-hide::-webkit-scrollbar-thumb,
div.gradio-container .summary-box::-webkit-scrollbar-thumb,
div.gradio-container .summary-box-tab2::-webkit-scrollbar-thumb,
div.gradio-container .scroll-hide::-webkit-scrollbar-thumb {
    background: #bdc1c6 !important;
    border-radius: 4px !important;
}

div.gradio-container .source-box textarea::-webkit-scrollbar-thumb:hover,
div.gradio-container .source-box textarea.scroll-hide::-webkit-scrollbar-thumb:hover,
div.gradio-container .summary-box::-webkit-scrollbar-thumb:hover,
div.gradio-container .summary-box-tab2::-webkit-scrollbar-thumb:hover,
div.gradio-container .scroll-hide::-webkit-scrollbar-thumb:hover {
    background: #9aa0a6 !important;
}

/* Summary Box in Tab 1 */
.summary-box.block {
    background: #f8f9fa !important;
}

.summary-box.block > .prose,
.summary-box.block > div {
    height: 100% !important;
    width: 100% !important;
    padding: 14px 16px !important;
    margin: 0 !important;
    border: none !important;
    box-sizing: border-box !important;
    overflow-y: auto !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #202124 !important;
    scrollbar-width: thin !important;
    scrollbar-color: #bdc1c6 #f8f9fa !important;
}

.summary-box.block * {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Tabs 2, 3, 4: Balanced Height Summary Card */
.summary-box-tab2.block {
    background: #f8f9fa !important;
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    min-height: 220px !important;
    max-height: 220px !important;
    height: 220px !important;
    overflow-y: auto !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #202124 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    margin: 0 0 10px 0 !important;
    box-sizing: border-box !important;
    scrollbar-width: thin !important;
    scrollbar-color: #bdc1c6 #f8f9fa !important;
}
.summary-box-tab2.block > .prose,
.summary-box-tab2.block > div {
    height: auto !important;
    min-height: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.summary-box-tab2.block *,
.summary-box.block * {
    background: transparent !important;
    outline: none !important;
}
.summary-box-tab2.block p,
.summary-box.block p {
    margin-bottom: 8px !important;
    line-height: 1.6 !important;
}
.summary-box-tab2.block ul,
.summary-box.block ul {
    margin: 6px 0 8px 18px !important;
    padding: 0 !important;
}
.summary-box-tab2.block li,
.summary-box.block li {
    margin-bottom: 5px !important;
    line-height: 1.5 !important;
}

/* Form Controls & Inputs */
input, select {
    border-radius: 6px !important;
    border: 1px solid #dadce0 !important;
    font-size: 13.5px !important;
    color: #202124 !important;
    background-color: #ffffff !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #1a73e8 !important;
    outline: none !important;
    box-shadow: 0 0 0 1px #1a73e8 !important;
}

/* File Upload Box */
.file-upload-box.block {
    min-height: 110px !important;
    border: 1px dashed #dadce0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

/* Metrics Table */
#metrics-card {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
}
#metrics-card table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 10px 0 !important;
}
#metrics-card th {
    background: #f8f9fa !important;
    color: #5f6368 !important;
    padding: 10px 14px !important;
    border: 1px solid #dadce0 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    text-align: left !important;
}
#metrics-card td {
    padding: 9px 14px !important;
    border: 1px solid #dadce0 !important;
    color: #3c4043 !important;
    font-size: 13px !important;
}

/* Accordions */
.gr-accordion {
    border: 1px solid #dadce0 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    margin-top: 14px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    overflow: hidden !important;
}
.gr-accordion .label-wrap {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #3c4043 !important;
    padding: 10px 16px !important;
}

/* Full Transcript Textbox - Controlled Height & Explicit Scrollbar */
.full-transcript-box textarea,
div.gradio-container .full-transcript-box textarea {
    height: 240px !important;
    max-height: 240px !important;
    min-height: 240px !important;
    overflow-y: scroll !important;
    overflow-x: hidden !important;
    scrollbar-width: thin !important;
    scrollbar-color: #bdc1c6 #f8f9fa !important;
    font-size: 13.5px !important;
    line-height: 1.6 !important;
    color: #202124 !important;
    background: #ffffff !important;
    padding: 12px 14px !important;
    border: 1px solid #dadce0 !important;
    border-radius: 6px !important;
    box-sizing: border-box !important;
}

div.gradio-container .full-transcript-box textarea::-webkit-scrollbar,
div.gradio-container .full-transcript-box textarea.scroll-hide::-webkit-scrollbar {
    display: block !important;
    width: 8px !important;
    background: #f1f3f4 !important;
}

div.gradio-container .full-transcript-box textarea::-webkit-scrollbar-track {
    background: #f1f3f4 !important;
    border-radius: 4px !important;
}

div.gradio-container .full-transcript-box textarea::-webkit-scrollbar-thumb {
    background: #bdc1c6 !important;
    border-radius: 4px !important;
}

div.gradio-container .full-transcript-box textarea::-webkit-scrollbar-thumb:hover {
    background: #9aa0a6 !important;
}
"""

SAMPLE_VI = (
    "Trí tuệ nhân tạo (AI) và các mô hình học sâu đang tạo ra một bước ngoặt lịch sử trong cuộc cách mạng công nghiệp lần thứ tư, "
    "tác động sâu rộng đến cơ cấu vận hành của nền kinh tế toàn cầu. Trong lĩnh vực y tế, các hệ thống thị giác máy tính tiên tiến đã "
    "hỗ trợ bác sĩ chẩn đoán hình ảnh X-quang, MRI với độ chính xác tương đương các chuyên gia đầu ngành, giúp phát hiện sớm các khối u "
    "ác tính và giảm thiểu đáng kể tỷ lệ sai sót y khoa. Song song đó, trong khối doanh nghiệp sản xuất và thương mại điện tử, các "
    "thuật toán tối ưu hóa thông minh đã tự động hóa việc dự báo nhu cầu thị trường, điều phối vận tải logistics và quản trị chuỗi "
    "cung ứng theo thời gian thực, tiết kiệm hàng tỷ USD chi phí vận hành mỗi năm.\n\n"
    "Tuy nhiên, tốc độ phổ cập vũ bão của AI cũng kéo theo những thách thức an ninh và pháp lý chưa từng có tiền lệ. Nguy cơ rò rỉ dữ liệu "
    "nhạy cảm, vi phạm quyền riêng tư của người dùng và sự xuất hiện của các lỗ hổng bảo mật trong mô hình đang trở thành bài toán hóc búa "
    "cho các nhà hoạch định chính sách. Đặc biệt, hiện tượng ảo giác thông tin và tính thiếu minh bạch trong các mô hình hộp đen đặt ra "
    "những câu hỏi hóc búa về trách nhiệm pháp lý khi xảy ra sai lệch trong các quyết định tự động. Trước tình hình đó, các tổ chức quốc tế "
    "và chính phủ nhiều nước đang khẩn trương thiết lập khung pháp lý toàn diện, yêu cầu các hệ thống AI thương mại phải đáp ứng tiêu "
    "chuẩn nghiêm ngặt về tính giải thích được, bảo mật dữ liệu và luôn đặt sự kiểm soát của con người làm trọng tâm tối thượng."
)

SAMPLE_EN = (
    "The transition from traditional statistical learning to deep transformer-based foundation models has fundamentally altered the paradigm "
    "of artificial intelligence. By utilizing self-attention mechanisms, modern neural networks capture long-range contextual dependencies "
    "across billions of multimodal tokens, achieving superhuman capabilities in code synthesis, language comprehension, and spatial reasoning. "
    "In enterprise environments, these models are rapidly evolving from static chatbots into autonomous workflow copilots capable of "
    "orchestrating complex business processes, automating financial reconciliations, and accelerating drug discovery pipelines.\n\n"
    "However, deploying foundation models in mission-critical production environments presents formidable architectural hurdles. Challenges "
    "such as stochastic hallucinations, significant GPU memory footprints, and non-deterministic reasoning trajectories necessitate rigorous "
    "verification frameworks. To mitigate these risks, cutting-edge AI architectures are transitioning toward agentic systems equipped with "
    "dynamic retrieval-augmented generation (RAG), sandboxed tool execution, and self-correcting reflection loops. The consensus across industry "
    "research labs emphasizes that sustainable AI deployment must balance scalable compute efficiency with robust safety guardrails and "
    "verifiable human oversight."
)

SAMPLE_FR = (
    "L'intelligence artificielle et l'apprentissage profond transforment en profondeur notre société et notre économie moderne. "
    "Dans le secteur médical, les algorithmes de vision par ordinateur assistent désormais les radiologues avec une précision remarquable "
    "pour détecter précocement les pathologies complexes, optimisant ainsi les protocoles de soins pour des millions de patients. "
    "Parallèlement, dans le domaine de la transition énergétique, les réseaux neuronaux permettent d'ajuster en temps réel la distribution "
    "des énergies renouvelables et d'optimiser l'efficacité thermique des infrastructures urbaines.\n\n"
    "Néanmoins, cette transformation numérique rapide soulève des interrogations éthiques et réglementaires majeures. "
    "La protection des données personnelles, la souveraineté numérique et la transparence des modèles d'apprentissage automatique "
    "constituent des enjeux cruciaux pour les institutions européennes. L'Union européenne a ainsi adopté l'AI Act afin d'encadrer "
    "strictement les systèmes à haut risque, d'imposer des audits rigoureux et de garantir que la technologie reste au service de l'humain."
)

SAMPLE_ES = (
    "La inteligencia artificial está revolucionando de manera acelerada múltiples sectores industriales y científicos en todo el mundo. "
    "En el ámbito de la biomedicina, las herramientas avanzadas de procesamiento de lenguaje natural y visión artificial permiten acelerar "
    "el descubrimiento de nuevos fármacos y analizar imágenes clínicas complejas con un nivel de exactitud sin precedentes, mejorando "
    "drásticamente el pronóstico y tratamiento de los pacientes. Asimismo, en el sector agroalimentario, los modelos predictivos optimizan "
    "el uso eficiente del agua và reducen los impactos ambientales en la producción agrícola.\n\n"
    "Sin embargo, el despliegue generalizado de estas soluciones autónomas plantea retos considerables en materia de privacidad, equidad "
    "algorítmica y seguridad digital. Los sesgos inherentes a los datos de entrenamiento y la opacidad de los algoritmos de caja negra exigen "
    "la formulación de normativas sólidas. Diversos organismos globales y gobiernos están promoviendo marcos regulatorios que garanticen "
    "la explicabilidad, la supervisión humana continua y el uso ético y responsable de la inteligencia artificial."
)

LANG_CHOICES = ["Auto", "Vietnamese", "English", "French", "Spanish"]

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="VideoInsight · Multilingual Video & Speech Summarizer", theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML(
        "<style>" + CUSTOM_CSS + "</style>\n"
        """<script>
        (function() {
            function resetSourceScroll() {
                var el = document.querySelector('.source-box textarea');
                if (el) {
                    el.scrollTop = 0;
                    try { el.setSelectionRange(0, 0); } catch(e) {}
                }
            }
            document.addEventListener('click', function(e) {
                var pill = e.target && (e.target.classList.contains('sample-pill') || e.target.closest('.sample-pill'));
                if (pill) {
                    [40, 100, 200, 350, 500, 800, 1200].forEach(function(t) {
                        setTimeout(resetSourceScroll, t);
                    });
                }
            });
        })();
        </script>"""
    )

    # Engineering Clean Header with Cohesive Visual Accents
    gr.HTML(
        """
        <div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px; font-weight: 700; color: #202124; letter-spacing: -0.02em;">VideoInsight</span>
                <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid #d2e3fc;">NLP Platform</span>
              </div>
              <div style="color: #5f6368; font-size: 13px; margin-top: 4px;">
                Hệ thống nhận diện giọng nói &amp; tóm tắt video đa ngôn ngữ (Vietnamese · English · French · Spanish)
              </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; font-size: 12px; color: #3c4043;">
              <span style="background: #f8f9fa; padding: 5px 12px; border-radius: 16px; font-weight: 500; border: 1px solid #dadce0;">Whisper ASR</span>
              <span style="background: #f8f9fa; padding: 5px 12px; border-radius: 16px; font-weight: 500; border: 1px solid #dadce0;">Fine-tuned mT5 (582M)</span>
              <span style="background: #e8f0fe; color: #1a73e8; padding: 5px 12px; border-radius: 16px; font-weight: 600; border: 1px solid #d2e3fc;">Hierarchical Map-Reduce</span>
            </div>
          </div>
        </div>
        """
    )

    # Full-width Top-Level Tabs (Order: Video/Audio -> Link URL -> Văn bản -> File Phụ Đề)
    with gr.Tabs():

        # =========================================================================
        # TAB 1: UPLOAD VIDEO / AUDIO (WHISPER ASR) - First Position per User Request
        # =========================================================================
        with gr.Tab("🎥 Video / Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div style='height: 28px; display: flex; align-items: center; font-size: 14px; font-weight: 600; color: #202124;'>Tải lên tệp Video hoặc Âm thanh</div>")
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="height: 28px; display: flex; align-items: center; justify-content: space-between; font-size: 14px; font-weight: 600; color: #202124;">
                          <span>Bản tóm tắt điều hành</span>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid #d2e3fc;">Executive Summary</span>
                        </div>
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    video_in = gr.File(
                        label="Tệp phương tiện (.mp4, .mkv, .avi, .mov, .mp3, .wav, .m4a)",
                        file_types=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a"],
                        type="filepath",
                        elem_classes=["file-upload-box"]
                    )
                    with gr.Row():
                        lang_opt_3 = gr.Dropdown(
                            choices=LANG_CHOICES,
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        whisper_opt_3 = gr.Dropdown(
                            choices=["tiny", "base", "small"],
                            value="base",
                            label="Whisper ASR",
                            scale=1
                        )
                    btn_video = gr.Button("Xử lý Video / Audio", variant="primary")

                    with gr.Accordion("Tùy chọn tham số sinh", open=False):
                        sl_max_3 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_3 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_3 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    exec_out_3 = gr.Markdown(
                        value="*Kết quả nhận diện giọng nói và tóm tắt sẽ xuất hiện tại đây.*",
                        elem_classes=["summary-box-tab2"]
                    )
                    download_out_3 = gr.DownloadButton(
                        label="Tải báo cáo tóm tắt (.md)",
                        value=None,
                        size="sm",
                        variant="secondary",
                        elem_classes=["dl-pill"]
                    )
                    meta_out_3 = gr.Markdown()

            with gr.Accordion("Phân đoạn theo dòng thời gian (Timeline & Highlights)", open=True):
                chapter_out_3 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("Toàn văn lời thoại transcript (Full Transcript)", open=False):
                transcript_out_3 = gr.Textbox(
                    label="Toàn bộ lời thoại nhận diện từ âm thanh video",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    elem_classes=["full-transcript-box"]
                )
            with gr.Accordion("Chỉ số đánh giá định lượng NLP (ROUGE Benchmark)", open=False):
                stats_out_3 = gr.Markdown(value="*Chỉ số ROUGE và đánh giá nén.*", elem_id="metrics-card")

        # =========================================================================
        # TAB 2: LINK VIDEO (URL) - Second Position
        # =========================================================================
        with gr.Tab("🔗 Link Video (URL)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div style='height: 28px; display: flex; align-items: center; font-size: 14px; font-weight: 600; color: #202124;'>Đường dẫn video YouTube / Trực tuyến</div>")
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="height: 28px; display: flex; align-items: center; justify-content: space-between; font-size: 14px; font-weight: 600; color: #202124;">
                          <span>Bản tóm tắt điều hành</span>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid #d2e3fc;">Executive Summary</span>
                        </div>
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    url_in = gr.Textbox(
                        placeholder="Dán đường dẫn video YouTube vào đây (ví dụ: https://www.youtube.com/watch?v=...)",
                        lines=2,
                        max_lines=2,
                        show_label=False
                    )

                    gr.HTML("<div style='font-size: 12px; font-weight: 500; color: #5f6368; margin-top: 4px; margin-bottom: 4px;'>Video mẫu thử nghiệm (4 ngôn ngữ chính thức):</div>")
                    with gr.Row():
                        btn_url_vi = gr.Button("🇻🇳 Tiếng Việt: Nick Vujicic", size="sm", elem_classes=["sample-pill-lang"])
                        btn_url_en = gr.Button("🇬🇧 English: Steve Jobs", size="sm", elem_classes=["sample-pill-lang"])
                    with gr.Row():
                        btn_url_fr = gr.Button("🇫🇷 French: Mythes sur l'IA", size="sm", elem_classes=["sample-pill-lang"])
                        btn_url_es = gr.Button("🇪🇸 Spanish: Inteligencia Artificial", size="sm", elem_classes=["sample-pill-lang"])

                    with gr.Row():
                        lang_opt_2 = gr.Dropdown(
                            choices=LANG_CHOICES,
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        whisper_opt_2 = gr.Dropdown(
                            choices=["tiny", "base", "small"],
                            value="base",
                            label="Whisper ASR",
                            scale=1
                        )
                    btn_url = gr.Button("Tóm tắt từ Video URL", variant="primary")

                    with gr.Accordion("Tùy chọn tham số sinh", open=False):
                        sl_max_2 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_2 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_2 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    exec_out_2 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích, đầy đủ ý chính sẽ xuất hiện tại đây sau khi xử lý.*",
                        elem_classes=["summary-box-tab2"]
                    )
                    download_out_2 = gr.DownloadButton(
                        label="Tải báo cáo tóm tắt (.md)",
                        value=None,
                        size="sm",
                        variant="secondary",
                        elem_classes=["dl-pill"]
                    )
                    meta_out_2 = gr.Markdown()

            with gr.Accordion("Phân đoạn theo dòng thời gian (Timeline & Highlights)", open=True):
                chapter_out_2 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("Toàn văn lời thoại transcript (Full Transcript)", open=False):
                transcript_out_2 = gr.Textbox(
                    label="Toàn bộ lời thoại trích xuất từ audio / video / phụ đề (Full Transcript - Không cắt bớt)",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    elem_classes=["full-transcript-box"]
                )
            with gr.Accordion("Chỉ số đánh giá định lượng NLP (ROUGE Benchmark)", open=False):
                stats_out_2 = gr.Markdown(value="*Bảng chỉ số đo lường ROUGE, tỷ lệ nén và mức độ tóm tắt trừu tượng (Abstractiveness).*", elem_id="metrics-card")

        # =========================================================================
        # TAB 3: VĂN BẢN (TEXT) - Perfectly Symmetrical Google Translate Layout
        # =========================================================================
        with gr.Tab("文A Văn bản"):
            # Header row for left & right
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div style='height: 28px; display: flex; align-items: center; font-size: 14px; font-weight: 600; color: #202124;'>Văn bản nguồn (Source Text)</div>")
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="height: 28px; display: flex; align-items: center; justify-content: space-between; font-size: 14px; font-weight: 600; color: #202124;">
                          <span>Bản tóm tắt điều hành</span>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid #d2e3fc;">Executive Summary</span>
                        </div>
                        """
                    )

            # Main content boxes: Exact side-by-side equal height (300px)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    text_in = gr.Textbox(
                        placeholder="Nhập hoặc dán nội dung văn bản transcript cần tóm tắt, hoặc chọn dữ liệu mẫu bên dưới...",
                        lines=12,
                        max_lines=12,
                        show_label=False,
                        elem_classes=["source-box"]
                    )
                with gr.Column(scale=1):
                    exec_out_1 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích, mạch lạc sẽ xuất hiện tại đây sau khi xử lý.*",
                        elem_classes=["summary-box"]
                    )

            # Secondary action row: 4 equal sample buttons on left (25% each), Download button on right
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 12px; font-weight: 500; color: #5f6368; margin-top: 6px; margin-bottom: 4px;'>Dữ liệu mẫu thử nghiệm (Click để chọn nhanh):</div>")
                    with gr.Row(elem_classes=["sample-row"]):
                        btn_sample_vi = gr.Button("Vietnamese", size="sm", elem_classes=["sample-pill"])
                        btn_sample_en = gr.Button("English", size="sm", elem_classes=["sample-pill"])
                        btn_sample_fr = gr.Button("French", size="sm", elem_classes=["sample-pill"])
                        btn_sample_es = gr.Button("Spanish", size="sm", elem_classes=["sample-pill"])
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 12px; font-weight: 500; color: #5f6368; margin-top: 6px; margin-bottom: 4px;'>Xuất kết quả báo cáo:</div>")
                    download_out_1 = gr.DownloadButton(
                        label="Tải báo cáo tóm tắt (.md)",
                        value=None,
                        size="sm",
                        variant="secondary",
                        elem_classes=["dl-pill"]
                    )

            # Primary action row: Language option + Summarize CTA Button
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        lang_opt_1 = gr.Dropdown(
                            choices=LANG_CHOICES,
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        btn_text = gr.Button("Tóm tắt văn bản", variant="primary", scale=2)
                with gr.Column(scale=1):
                    meta_out_1 = gr.Markdown()

            with gr.Accordion("Tùy chọn tham số sinh (Hyperparameters)", open=False):
                sl_max_1 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                sl_min_1 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                sl_beam_1 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

        # =========================================================================
        # TAB 4: FILE PHỤ ĐỀ (.srt / .txt) - Fourth Position
        # =========================================================================
        with gr.Tab("📄 File Phụ Đề (.srt)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div style='height: 28px; display: flex; align-items: center; font-size: 14px; font-weight: 600; color: #202124;'>Tải lên tệp phụ đề (.srt, .vtt, .txt)</div>")
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="height: 28px; display: flex; align-items: center; justify-content: space-between; font-size: 14px; font-weight: 600; color: #202124;">
                          <span>Bản tóm tắt điều hành</span>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; border: 1px solid #d2e3fc;">Executive Summary</span>
                        </div>
                        """
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    srt_in = gr.File(
                        label="Tệp phụ đề (.srt, .vtt, .txt)",
                        file_types=[".srt", ".vtt", ".txt"],
                        type="filepath",
                        elem_classes=["file-upload-box"]
                    )
                    lang_opt_4 = gr.Dropdown(
                        choices=LANG_CHOICES,
                        value="Auto",
                        label="Ngôn ngữ tóm tắt"
                    )
                    btn_srt = gr.Button("Xử lý tệp phụ đề", variant="primary")

                    with gr.Accordion("Tùy chọn tham số sinh", open=False):
                        sl_max_4 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_4 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_4 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    exec_out_4 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích sẽ hiển thị tại đây.*",
                        elem_classes=["summary-box-tab2"]
                    )
                    download_out_4 = gr.DownloadButton(
                        label="Tải báo cáo tóm tắt (.md)",
                        value=None,
                        size="sm",
                        variant="secondary",
                        elem_classes=["dl-pill"]
                    )
                    meta_out_4 = gr.Markdown()

            with gr.Accordion("Phân đoạn theo dòng thời gian (Timeline & Highlights)", open=True):
                chapter_out_4 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("Toàn văn lời thoại transcript (Full Transcript)", open=False):
                transcript_out_4 = gr.Textbox(
                    label="Toàn bộ lời thoại trích xuất từ file phụ đề",
                    lines=8,
                    max_lines=12,
                    interactive=False,
                    elem_classes=["full-transcript-box"]
                )
            with gr.Accordion("Chỉ số đánh giá định lượng NLP (ROUGE Benchmark)", open=False):
                stats_out_4 = gr.Markdown(value="*Chỉ số ROUGE và đánh giá nén.*", elem_id="metrics-card")

    # =========================================================================
    # EVENT BINDINGS
    # =========================================================================

    # Tab 1: Text
    btn_text.click(
        fn=run_text_tab,
        inputs=[text_in, lang_opt_1, sl_max_1, sl_min_1, sl_beam_1],
        outputs=[meta_out_1, exec_out_1, download_out_1],
        api_name="summarize_text"
    )
    RESET_SCROLL_JS = "() => { setTimeout(() => { const el = document.querySelector('.source-box textarea'); if (el) { el.scrollTop = 0; try { el.setSelectionRange(0, 0); } catch(e) {} } }, 80); setTimeout(() => { const el = document.querySelector('.source-box textarea'); if (el) { el.scrollTop = 0; } }, 350); }"

    btn_sample_vi.click(fn=lambda: (SAMPLE_VI, "Vietnamese"), outputs=[text_in, lang_opt_1], js=RESET_SCROLL_JS)
    btn_sample_en.click(fn=lambda: (SAMPLE_EN, "English"), outputs=[text_in, lang_opt_1], js=RESET_SCROLL_JS)
    btn_sample_fr.click(fn=lambda: (SAMPLE_FR, "French"), outputs=[text_in, lang_opt_1], js=RESET_SCROLL_JS)
    btn_sample_es.click(fn=lambda: (SAMPLE_ES, "Spanish"), outputs=[text_in, lang_opt_1], js=RESET_SCROLL_JS)

    # Tab 2: URL
    btn_url.click(
        fn=run_url_tab,
        inputs=[url_in, lang_opt_2, whisper_opt_2, sl_max_2, sl_min_2, sl_beam_2],
        outputs=[meta_out_2, exec_out_2, chapter_out_2, stats_out_2, transcript_out_2, download_out_2],
        api_name="summarize_url"
    )
    btn_url_vi.click(fn=lambda: ("https://www.youtube.com/watch?v=cOsG32MuQWQ", "Vietnamese"), outputs=[url_in, lang_opt_2])
    btn_url_en.click(fn=lambda: ("https://www.youtube.com/watch?v=UF8uR6Z6KLc", "English"), outputs=[url_in, lang_opt_2])
    btn_url_fr.click(fn=lambda: ("https://www.youtube.com/watch?v=8JY_TgCO4Fw", "French"), outputs=[url_in, lang_opt_2])
    btn_url_es.click(fn=lambda: ("https://www.youtube.com/watch?v=znq3ql6wqnE", "Spanish"), outputs=[url_in, lang_opt_2])

    # Tab 3: Video / Audio
    btn_video.click(
        fn=run_video_tab,
        inputs=[video_in, lang_opt_3, whisper_opt_3, sl_max_3, sl_min_3, sl_beam_3],
        outputs=[meta_out_3, exec_out_3, chapter_out_3, stats_out_3, transcript_out_3, download_out_3],
        api_name="summarize_video"
    )

    # Tab 4: SRT File
    btn_srt.click(
        fn=run_srt_tab,
        inputs=[srt_in, lang_opt_4, sl_max_4, sl_min_4, sl_beam_4],
        outputs=[meta_out_4, exec_out_4, chapter_out_4, stats_out_4, transcript_out_4, download_out_4],
        api_name="summarize_srt"
    )

if __name__ == "__main__":
    demo.queue().launch()
