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
            "Tiếng Việt": "vi",
            "English": "en",
            "Français": "fr",
            "Español": "es",
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
                        f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(60,64,67,0.06);">
  <div style="margin-bottom: 6px;"><span style="background: #e8f0fe; color: #1a73e8; font-weight: 600; font-size: 12px; padding: 3px 10px; border-radius: 12px; display: inline-block;">Phân Đoạn {ch['chunk_id']} · ⏱️ {time_range}</span></div>
  <div style="font-size: 14.5px; color: #202124; line-height: 1.6;">{ch['summary']}</div>
</div>"""
                    )
                elif time_range and time_range != "Toàn văn":
                    chapter_md.append(
                        f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(60,64,67,0.06);">
  <div style="margin-bottom: 6px;"><span style="background: #e8f0fe; color: #1a73e8; font-weight: 600; font-size: 12px; padding: 3px 10px; border-radius: 12px; display: inline-block;">{time_range}</span></div>
  <div style="font-size: 14.5px; color: #202124; line-height: 1.6;">{ch['summary']}</div>
</div>"""
                    )
        elif chapters and any(re.search(r"\d{1,2}:\d{2}", str(c.get("time_str", ""))) for c in chapters):
            ch = chapters[0]
            chapter_md.append(
                f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(60,64,67,0.06);">
  <div style="margin-bottom: 6px;"><span style="background: #e8f0fe; color: #1a73e8; font-weight: 600; font-size: 12px; padding: 3px 10px; border-radius: 12px; display: inline-block;">Phân Đoạn 1 · ⏱️ {ch.get('time_str', '')}</span></div>
  <div style="font-size: 14.5px; color: #202124; line-height: 1.6;">{ch['summary']}</div>
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
                        f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(60,64,67,0.06);">
  <div style="margin-bottom: 6px;"><span style="background: #e8f0fe; color: #1a73e8; font-weight: 600; font-size: 12px; padding: 3px 10px; border-radius: 12px; display: inline-block;">{pt_title}</span></div>
  <div style="font-size: 14.5px; color: #202124; line-height: 1.6;">{content}</div>
</div>"""
                    )
                chapter_display = "\n".join(pts_md)
            else:
                chapter_display = f"> 📄 *Toàn bộ nội dung ({word_count} từ) đã được tóm tắt súc tích trong Bản Tóm Tắt ở trên.*"

        total_elapsed = time.time() - t_start

        # Clean Google-style Metadata Badge
        meta_info = (
            f"""<div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 8px 14px; margin-bottom: 10px; font-size: 13px; color: #5f6368; display: flex; flex-wrap: wrap; gap: 14px; align-items: center;">
              <span>📁 <strong>{title}</strong></span>
              <span>⏱️ {duration_str}</span>
              <span>🌐 <span style="background: #e8f0fe; color: #1a73e8; padding: 2px 8px; border-radius: 10px; font-weight: 600; font-size: 12px;">{detected_lang.upper()}</span></span>
              <span>📝 {word_count:,} từ</span>
              <span>⚡ <strong style="color: #137333;">{total_elapsed:.1f}s</strong></span>
              <span style="color: #1a73e8; font-weight: 500;">✓ CPU 16GB</span>
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
/* Google Material 3 Minimalist Clean Theme */
body, .gradio-container {
    background-color: #f8fafd !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Google Sans", Helvetica, Arial, sans-serif !important;
    color: #202124 !important;
}

.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 16px 20px !important;
}

/* Tabs: Full-width Top Navigation */
.tab-nav {
    border-bottom: 2px solid #e8eaed !important;
    background: transparent !important;
    gap: 6px !important;
    padding: 0 4px !important;
    margin-bottom: 16px !important;
}
.tab-nav button {
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    color: #5f6368 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 18px !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover {
    background: #f1f3f4 !important;
    color: #202124 !important;
}
.tab-nav button.selected {
    color: #1a73e8 !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #1a73e8 !important;
    background: transparent !important;
}

/* Primary Action Buttons: Google Blue Pill */
button.primary, .gr-button-primary {
    background: #1a73e8 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    box-shadow: 0 1px 3px rgba(60,64,67,0.3) !important;
    transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: #1765cc !important;
    box-shadow: 0 2px 6px rgba(60,64,67,0.4) !important;
}
button.primary:active {
    background: #185abc !important;
}

/* Output Result Card: Blue Top Border & Distinct Surface */
.summary-result-card {
    background: #ffffff !important;
    border: 1.5px solid #bfdbfe !important;
    border-top: 4px solid #1a73e8 !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    min-height: 270px !important;
    font-size: 15px !important;
    line-height: 1.75 !important;
    color: #202124 !important;
    box-shadow: 0 1px 4px rgba(60,64,67,0.08) !important;
}

/* Metrics Table */
#metrics-card {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
}
#metrics-card table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 10px 0 !important;
}
#metrics-card th {
    background: #f8f9fa !important;
    color: #3c4043 !important;
    padding: 10px 14px !important;
    border: 1px solid #dadce0 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}
#metrics-card td {
    padding: 9px 14px !important;
    border: 1px solid #dadce0 !important;
    color: #3c4043 !important;
    font-size: 13.5px !important;
}

/* Accordions */
.gr-accordion {
    border: 1px solid #dadce0 !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    margin-top: 12px !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06) !important;
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

with gr.Blocks(title="VideoInsight · AI Video & Subtitle Summarizer", css=CUSTOM_CSS) as demo:

    # Google Material Clean Header
    gr.HTML(
        """
        <div style="background: #ffffff; border: 1px solid #dadce0; border-radius: 12px; padding: 14px 22px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
          <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: #e8f0fe; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
              🎬
            </div>
            <div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px; font-weight: 700; color: #1a73e8; letter-spacing: -0.3px;">VideoInsight</span>
                <span style="background: #e8f0fe; color: #1a73e8; font-size: 11.5px; font-weight: 600; padding: 2px 8px; border-radius: 12px;">AI Summarizer</span>
              </div>
              <p style="margin: 2px 0 0 0; color: #5f6368; font-size: 13px;">
                Tóm tắt thông minh lời thoại video & văn bản đa ngữ (Vi · En · Fr · Es) với OpenAI Whisper & fine-tuned mT5
              </p>
            </div>
          </div>
          <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <span style="background: #f1f3f4; color: #3c4043; padding: 4px 10px; border-radius: 14px; font-size: 12px; font-weight: 500;">🎙️ Whisper ASR</span>
            <span style="background: #f1f3f4; color: #3c4043; padding: 4px 10px; border-radius: 14px; font-size: 12px; font-weight: 500;">🧠 mT5 Seq2Seq</span>
            <span style="background: #f1f3f4; color: #3c4043; padding: 4px 10px; border-radius: 14px; font-size: 12px; font-weight: 500;">⚡ Map-Reduce</span>
            <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 10px; border-radius: 14px; font-size: 12px; font-weight: 600;">☁️ High-Memory CPU</span>
          </div>
        </div>
        """
    )

    # Full-width Top-Level Tabs
    with gr.Tabs():

        # =========================================================================
        # TAB 1: 📝 VĂN BẢN (TEXT) - Pure Side-by-Side Summarization (No bottom clutter)
        # =========================================================================
        with gr.Tab("📝 1. Văn bản (Text)"):
            with gr.Row(equal_height=False):
                # Left Column: Input
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 13.5px; font-weight: 700; color: #475569; margin-bottom: 8px;'>📥 VĂN BẢN NGUỒN (INPUT TRANSCRIPT)</div>")
                    text_in = gr.Textbox(
                        label="",
                        placeholder="Nhập hoặc dán nội dung văn bản transcript vào đây, hoặc click chọn nhanh mẫu 4 ngôn ngữ bên dưới...",
                        lines=11
                    )
                    
                    gr.Markdown("<div style='font-size: 12.5px; font-weight: 600; color: #5f6368; margin-top: 4px; margin-bottom: 2px;'>💡 Chọn nhanh dữ liệu mẫu 4 ngôn ngữ:</div>")
                    with gr.Row():
                        btn_sample_vi = gr.Button("🇻🇳 Tiếng Việt", size="sm")
                        btn_sample_en = gr.Button("🇬🇧 English", size="sm")
                        btn_sample_fr = gr.Button("🇫🇷 Français", size="sm")
                        btn_sample_es = gr.Button("🇪🇸 Español", size="sm")

                    with gr.Row():
                        lang_opt_1 = gr.Dropdown(
                            choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        btn_text = gr.Button("Tóm tắt văn bản", variant="primary", scale=1)

                    with gr.Accordion("⚙️ Tùy chọn tham số", open=False):
                        sl_max_1 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_1 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_1 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                # Right Column: Output
                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                          <div style="font-size: 13.5px; font-weight: 700; color: #1a73e8;">✨ BẢN TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</div>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;">Output Card</span>
                        </div>
                        """
                    )
                    meta_out_1 = gr.Markdown()
                    exec_out_1 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích, đầy đủ ý chính sẽ xuất hiện tại đây sau khi bạn nhấn nút tóm tắt.*",
                        elem_classes=["summary-result-card"]
                    )
                    download_out_1 = gr.File(
                        label="📥 Tải báo cáo tóm tắt (.md)",
                        interactive=False
                    )

        # =========================================================================
        # TAB 2: 📁 FILE PHỤ ĐỀ (.srt / .txt)
        # =========================================================================
        with gr.Tab("📁 2. File Phụ Đề (.srt / .txt)"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 13.5px; font-weight: 700; color: #475569; margin-bottom: 8px;'>📁 TẢI LÊN FILE PHỤ ĐỀ</div>")
                    srt_in = gr.File(
                        label="File phụ đề (.srt, .vtt, .txt)",
                        file_types=[".srt", ".vtt", ".txt"],
                        type="filepath"
                    )
                    with gr.Row():
                        lang_opt_2 = gr.Dropdown(
                            choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        btn_srt = gr.Button("Tóm tắt phụ đề", variant="primary", scale=1)

                    with gr.Accordion("⚙️ Tùy chọn tham số", open=False):
                        sl_max_2 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_2 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_2 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                          <div style="font-size: 13.5px; font-weight: 700; color: #1a73e8;">✨ BẢN TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</div>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;">Output Card</span>
                        </div>
                        """
                    )
                    meta_out_2 = gr.Markdown()
                    exec_out_2 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích sẽ hiển thị tại đây.*",
                        elem_classes=["summary-result-card"]
                    )
                    download_out_2 = gr.File(
                        label="📥 Tải báo cáo tóm tắt (.md)",
                        interactive=False
                    )

            with gr.Accordion("📍 Dòng Thời Gian & Phân Đoạn (Chapter Timeline)", open=True):
                chapter_out_2 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("📄 Lời Thoại Transcript Đầy Đủ 100% (Full Transcript)", open=False):
                transcript_out_2 = gr.Textbox(label="Toàn bộ lời thoại trích xuất từ file phụ đề", lines=10, max_lines=30, interactive=False)
            with gr.Accordion("📊 Đánh Giá Đo Lường NLP (ROUGE Metrics)", open=False):
                stats_out_2 = gr.Markdown(value="*Chỉ số ROUGE và đánh giá nén.*", elem_id="metrics-card")

        # =========================================================================
        # TAB 3: 🎬 UPLOAD VIDEO / AUDIO (WHISPER ASR)
        # =========================================================================
        with gr.Tab("🎬 3. Upload Video / Audio"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 13.5px; font-weight: 700; color: #475569; margin-bottom: 8px;'>🎙️ TẢI LÊN FILE VIDEO / AUDIO</div>")
                    video_in = gr.File(
                        label="File Video hoặc Âm thanh (.mp4, .mp3, .wav, .m4a)",
                        file_types=[".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav", ".m4a"],
                        type="filepath"
                    )
                    with gr.Row():
                        lang_opt_3 = gr.Dropdown(
                            choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
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
                    btn_video = gr.Button("Tóm tắt Video/Audio", variant="primary")

                    with gr.Accordion("⚙️ Tùy chọn tham số sinh", open=False):
                        sl_max_3 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_3 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_3 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                          <div style="font-size: 13.5px; font-weight: 700; color: #1a73e8;">✨ BẢN TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</div>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;">Output Card</span>
                        </div>
                        """
                    )
                    meta_out_3 = gr.Markdown()
                    exec_out_3 = gr.Markdown(
                        value="*Kết quả nhận diện giọng nói và tóm tắt sẽ xuất hiện tại đây.*",
                        elem_classes=["summary-result-card"]
                    )
                    download_out_3 = gr.File(
                        label="📥 Tải báo cáo tóm tắt (.md)",
                        interactive=False
                    )

            with gr.Accordion("📍 Dòng Thời Gian & Phân Đoạn Video (Chapter Timeline)", open=True):
                chapter_out_3 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("📄 Lời Thoại Transcript Đầy Đủ 100% (Full Speech Transcript)", open=False):
                transcript_out_3 = gr.Textbox(label="Toàn bộ lời thoại nhận diện từ âm thanh video", lines=10, max_lines=30, interactive=False)
            with gr.Accordion("📊 Đánh Giá Đo Lường NLP (ROUGE Metrics)", open=False):
                stats_out_3 = gr.Markdown(value="*Chỉ số ROUGE và đánh giá nén.*", elem_id="metrics-card")

        # =========================================================================
        # TAB 4: 🔗 LINK VIDEO TRỰC TUYẾN (URL)
        # =========================================================================
        with gr.Tab("🔗 4. Link Video Trực Tuyến (URL)"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML("<div style='font-size: 13.5px; font-weight: 700; color: #475569; margin-bottom: 8px;'>🔗 LINK VIDEO TRỰC TUYẾN (YOUTUBE / URL)</div>")
                    url_in = gr.Textbox(
                        label="",
                        placeholder="Dán link YouTube vào đây (ví dụ: https://www.youtube.com/watch?v=...)",
                        lines=2
                    )

                    gr.Markdown("<div style='font-size: 12.5px; font-weight: 600; color: #5f6368; margin-top: 4px; margin-bottom: 2px;'>💡 Video mẫu có sẵn phụ đề chuẩn CC (Click để chọn nhanh):</div>")
                    with gr.Row():
                        btn_url_vi = gr.Button("🇻🇳 Video VN: Lúa & Khí Metan", size="sm")
                        btn_url_en = gr.Button("🇬🇧 Video EN: Steve Jobs Stanford Speech", size="sm")

                    with gr.Row():
                        lang_opt_4 = gr.Dropdown(
                            choices=["Auto", "Tiếng Việt", "English", "Français", "Español"],
                            value="Auto",
                            label="Ngôn ngữ tóm tắt",
                            scale=1
                        )
                        whisper_opt_4 = gr.Dropdown(
                            choices=["tiny", "base", "small"],
                            value="base",
                            label="Whisper ASR",
                            scale=1
                        )
                    btn_url = gr.Button("Tóm tắt từ Link URL", variant="primary")

                    with gr.Accordion("⚙️ Tùy chọn tham số sinh", open=False):
                        sl_max_4 = gr.Slider(80, 400, value=250, step=20, label="Max Tokens")
                        sl_min_4 = gr.Slider(20, 150, value=35, step=5, label="Min Tokens")
                        sl_beam_4 = gr.Slider(2, 8, value=4, step=1, label="Num Beams")

                with gr.Column(scale=1):
                    gr.HTML(
                        """
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                          <div style="font-size: 13.5px; font-weight: 700; color: #1a73e8;">✨ BẢN TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</div>
                          <span style="background: #e8f0fe; color: #1a73e8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;">Output Card</span>
                        </div>
                        """
                    )
                    meta_out_4 = gr.Markdown()
                    exec_out_4 = gr.Markdown(
                        value="*Kết quả tóm tắt súc tích, đầy đủ ý chính sẽ xuất hiện tại đây sau khi bạn nhấn nút tóm tắt.*",
                        elem_classes=["summary-result-card"]
                    )
                    download_out_4 = gr.File(
                        label="📥 Tải báo cáo tóm tắt (.md)",
                        interactive=False
                    )

            with gr.Accordion("📍 Dòng Thời Gian & Phân Đoạn Video (Chapter Timeline & Highlights)", open=True):
                chapter_out_4 = gr.Markdown(value="*Dòng thời gian các phân đoạn nội dung chính sẽ hiển thị tại đây.*")
            with gr.Accordion("📄 Lời Thoại Transcript Đầy Đủ 100% (Full Speech Transcript)", open=False):
                transcript_out_4 = gr.Textbox(label="Toàn bộ lời thoại trích xuất từ audio / video / phụ đề (Full Transcript - Không cắt bớt)", lines=12, max_lines=35, interactive=False)
            with gr.Accordion("📊 Đánh Giá Đo Lường NLP (ROUGE Metrics & Benchmark)", open=False):
                stats_out_4 = gr.Markdown(value="*Bảng chỉ số đo lường ROUGE, tỷ lệ nén và mức độ tóm tắt trừu tượng (Abstractiveness).*", elem_id="metrics-card")

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
    btn_sample_vi.click(fn=lambda: (SAMPLE_VI, "Tiếng Việt"), outputs=[text_in, lang_opt_1])
    btn_sample_en.click(fn=lambda: (SAMPLE_EN, "English"), outputs=[text_in, lang_opt_1])
    btn_sample_fr.click(fn=lambda: (SAMPLE_FR, "Français"), outputs=[text_in, lang_opt_1])
    btn_sample_es.click(fn=lambda: (SAMPLE_ES, "Español"), outputs=[text_in, lang_opt_1])

    # Tab 2: SRT File
    btn_srt.click(
        fn=run_srt_tab,
        inputs=[srt_in, lang_opt_2, sl_max_2, sl_min_2, sl_beam_2],
        outputs=[meta_out_2, exec_out_2, chapter_out_2, stats_out_2, transcript_out_2, download_out_2],
        api_name="summarize_srt"
    )

    # Tab 3: Video / Audio
    btn_video.click(
        fn=run_video_tab,
        inputs=[video_in, lang_opt_3, whisper_opt_3, sl_max_3, sl_min_3, sl_beam_3],
        outputs=[meta_out_3, exec_out_3, chapter_out_3, stats_out_3, transcript_out_3, download_out_3],
        api_name="summarize_video"
    )

    # Tab 4: URL
    btn_url.click(
        fn=run_url_tab,
        inputs=[url_in, lang_opt_4, whisper_opt_4, sl_max_4, sl_min_4, sl_beam_4],
        outputs=[meta_out_4, exec_out_4, chapter_out_4, stats_out_4, transcript_out_4, download_out_4],
        api_name="summarize_url"
    )
    btn_url_vi.click(fn=lambda: ("https://www.youtube.com/watch?v=VAIe6L68M3E", "Tiếng Việt"), outputs=[url_in, lang_opt_4])
    btn_url_en.click(fn=lambda: ("https://www.youtube.com/watch?v=UF8uR6Z6KLc", "English"), outputs=[url_in, lang_opt_4])

if __name__ == "__main__":
    demo.queue().launch()
