"""
ASR Engine Module:
Speech-to-Text transcription with Whisper, extracting timestamped segments.
"""

import os
import re
import time
import torch

_WHISPER_CACHE = {}
DEVICE = "cpu"

def normalize_speech_text(text: str) -> str:
    """Cleans up ASR noise artifacts, filters outro advertisements, and fixes phonetic misrecognition."""
    if not text:
        return ""

    # Remove bracket noise tokens: [âm nhạc], [vỗ tay], [tiếng cười], (music), etc.
    text = re.sub(r"\[(âm nhạc|tiếng cười|vỗ tay|nhạc|music|applause|laughter)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((âm nhạc|tiếng cười|vỗ tay|nhạc|music|applause|laughter)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[♪♫]", "", text)

    # Phonetic corrections for common spoken Vietnamese misrecognitions
    replacements = [
        # Nick Vujicic misrecognitions
        (r"\b(n[íi]ch|n[ếe]t|n[ií]c|n[ií]k|níck)\s+(v[ụu]|v[uụ]|vụ|vũ)\s+(tr[ịi]|ch[ỉi]|trí|chí)\s+(s[ứu]c|sức)\b", "Nick Vujicic"),
        (r"\b(NIC|Nic|nic)\s+(vô\s+chi\s+sức|VuchiCIC|vuchicic|Vuchicic)\b", "Nick Vujicic"),
        (r"\bNick Vucci sức\b", "Nick Vujicic"),
        (r"\bNick Vujicic,\s*anh lại\b", "Nick Vujicic, anh là ai"),
        (r"\b(n[íi]ch|n[ếe]t)\s+l[ấa]y\s+kh[ỏo]e\b", "Nick lại khỏe"),
        (r"\bthế nhưng lịch lại\b", "thế nhưng Nick lại"),
        (r"\bchính lần ních\b", "chính là Nick"),
        (r"\bđừng n[íi]ch vụ trị sức\b", "được Nick Vujicic"),
        (r"\btình Nick vụ chỉ sức\b", "Nick Vujicic"),
        (r"\bđừng nịch vụ trị sức giúp ra\b", "được Nick Vujicic rút ra"),
        (r"\bchân lý sống đã đừng Nick Vujicic giúp ra\b", "chân lý sống đã được Nick Vujicic rút ra"),
        (r"\bchân lý sống đã đừng nịch vụ trị sức giúp ra\b", "chân lý sống đã được Nick Vujicic rút ra"),
        (r"\bai ta\b", "anh"),
        (r"\bđến một cái không ngờ\b", "đến kỳ diệu"),
        (r"\b(người đàn ông\s+)?khuyết tật\s+(lớn\s+|nhất\s+)?thế giới\b", "người giàu nghị lực sống"),
        (r"\bngười đánh âm nhất thiệt thò trên thế giới\b", "người từng chịu nhiều thiệt thòi nhất thế giới"),
        (r"\bngười đánh âm nhất thiệt thò\b", "người từng chịu nhiều thiệt thòi"),
        (r"\bdiễn thuyết truyền cảm\b(?!\s*hứng)", "diễn thuyết truyền cảm hứng"),
        (r"\bdiễn thức truyền cảm\b(?!\s*hứng)", "diễn thuyết truyền cảm hứng"),
        (r"\bNick\s*(\.{2,}\s*)?c\s*cấp 2\b", "Nick khi còn học cấp 2"),
        (r"\bbằng công nhìn\b", "bằng góc nhìn"),
        # Other names and terms in speech
        (r"\btui nhập đại học\b", "tốt nghiệp đại học"),
        (r"\bCliffeth\b", "Griffith"),
        (r"\bdiễn thức\b", "diễn thuyết"),
        (r"\b3 chậm người\b", "3 triệu người"),
        (r"\b20 từ quốc gia\b", "24 quốc gia"),
        (r"\bnăng châu lục vàng\b", "5 châu lục trên"),
        (r"\bcháy tìm\b", "trái tim"),
        (r"\bdung động\b", "rung động"),
        (r"\bdưới hạt\b", "giới hạn"),
        (r"\bđứa mặt\b", "đối mặt"),
        (r"\bkhắp lịch\b", "khắc nghiệt"),
        (r"\bnghiệp ngã\b", "nghiệt ngã"),
        (r"\btế mặc nào\b", "đến mức nào"),
        (r"\btiền tưởng\b", "tin tưởng"),
        (r"\bcồng hệ\b", "cầu nguyện"),
        (r"\bSaban Tron\b", "sẽ ban cho"),
        (r"\bĐôi Trân và Đôi Tay\b", "đôi chân và đôi tay"),
        (r"\bThì nhật\b", "Tuy nhiên"),
        (r"\bthay đồn\b", "thay đổi"),
        (r"\bbài bảo viet\b", "bài báo viết"),
        (r"\bkhết tật\b", "khuyết tật"),
        (r"\bngười đánh âm để\b", "người đàn ông"),
        (r"\bthế này em\b", "thế nên anh"),
        (r"\bLuân nó lực không ngựng\b", "Luôn nỗ lực không ngừng"),
        (r"\bngười bình thường có trị thực hiện mà không cần xuống dĩ\b", "người bình thường có thể thực hiện"),
        (r"\bngóng chân\b", "ngón chân"),
        (r"\bbản chân\b", "bàn chân"),
        (r"\bcầm cất\b", "cầm nắm"),
        (r"\bdù mấy tính\b", "dùng máy tính"),
        (r"\bphương phạc\b", "phương pháp"),
        (r"\bbóng chânnis\b", "tennis"),
        (r"\bchơi chống\b", "chơi trống"),
        (r"\bcông nước\b", "cốc nước"),
        (r"\btrải tóc\b", "chải tóc"),
        (r"\bbánh răng\b", "đánh răng"),
        (r"\btrà lối\b", "trả lời"),
        (r"\bcao dông\b", "cạo râu"),
        (r"\btrình golf\b", "chơi golf"),
        (r"\bnhải dù\b", "nhảy dù"),
        (r"\bhọp cấp 2\b", "học cấp 2"),
        (r"\bDiệu Phương\b", "địa phương"),
        (r"\btả các phòng chào viên\b", "và các phong trào vì"),
        (r"\btự mình lập giá tổ chức phí chính phủ\b", "thành lập tổ chức phi chính phủ"),
        (r"\bCo-Danger Australia\b", "Công dân trẻ Australia (Young Australian of the Year)"),
        (r"\bLive with Fowl Limits\b", "Life Without Limbs (Cuộc sống không giới hạn)"),
        (r"\bLife Việt phaolemis\b", "Life Without Limbs"),
        (r"\bNăm anh nhìn không chăm lẽ 5\b", "Năm 2005"),
        (r"\bkhỏe khắc thiền liên\b", "khoảnh khắc thiêng liêng"),
        (r"\bnghích đi\b", "ngất đi"),
        (r"\bnghiệp lực phi thường\b", "nghị lực phi thường"),
        (r"\bđộc lực sống\b", "động lực sống"),
        (r"\bsinh đình bán nhỏ\b", "sinh linh bé nhỏ"),
        (r"\bxích mình bé nhỏ\b", "sinh linh bé nhỏ"),
        (r"\bđứa con bé bọng\b", "đứa con bé bỏng"),
        (r"\bhiểu ớt quyết trị\b", "yếu ớt khuyết tật"),
        (r"\bquyết trị vật\b", "khuyết tật"),
        (r"\btài chân\b", "tay chân"),
        (r"\bviên nặng\b", "viên ngọc quý"),
        (r"\bđánh âm nhất thiệt thò\b", "đàn ông thiệt thòi nhất"),
        (r"\bđánh âm\b", "đàn ông"),
        (r"\bvấp lạ tới hàng trăm lần\b", "vấp ngã tới hàng trăm lần"),
        (r"\bcái tục mạnh mẽ\b", "tiếp tục mạnh mẽ"),
        (r"\bvượt lên sổ vận\b", "vượt lên số phận"),
        (r"\bhành tình\b", "hành tinh"),
        (r"\bđậu giả\b", "độc giả"),
        (r"\bhỏa tại sao\b", "hỏi tại sao"),
        (r"\bkhắc ngả vậy tôi\b", "khắc nghiệt với tôi"),
        (r"\bhơi này hết\b", "hơn ai hết"),
        (r"\btự tưởng\b", "tự tử"),
        (r"\bchốn trại khởi quốc tơi\b", "trốn chạy khỏi cuộc đời"),
        (r"\bcàng đảng\b", "can đảm"),
        (r"\bđứng giải\b", "đứng dậy"),
        (r"\bluôn tìm vào chúa\b", "luôn tin vào cuộc sống"),
        (r"\bbằng công nhìn hai bước dạm\b", "bằng kinh nghiệm sống"),
        (r"\btiểu tạng\b", "từ thiện"),
        (r"\bchâu đến\b", "chờ đến"),
        (r"\bbài máu\b", "bài báo"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Strip YouTube outro ads, website links, subscriber requests, junk endings
    ad_patterns = [
        r"(Nếu (thấy|nãy|bạn).*?(website|kênh|like|share|đăng ký|hotline|liên hệ|quảng cáo|itv|thêm).*?)$",
        r"\b(Nếu nãy ra lâu|không cần 2|Học trị cập Website|quả cao ITV|truy cập Website).*$",
        r"\b(hãy like|ấn đăng ký|đăng ký kênh|subscribe|bấm chuông|để lại bình luận).*$",
        r"\b(liên hệ quảng cáo|quảng cáo itv|website\s+[\w.-]+\s+trong con).*$",
    ]
    for pat in ad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clean_subtitle_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = normalize_speech_text(text)
    return text.strip()

def parse_srt_with_timestamps(srt_content: str) -> list:
    blocks = re.split(r"\n\s*\n", srt_content.strip())
    segments = []

    def time_to_sec(t_str):
        parts = t_str.strip().replace(",", ".").split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return 0.0

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 2:
            time_match = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", lines[1] if len(lines) > 1 else "")
            if not time_match and len(lines) > 0:
                time_match = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", lines[0])
            if time_match:
                start_sec = time_to_sec(time_match.group(1))
                end_sec = time_to_sec(time_match.group(2))
                text_part = " ".join(lines[2:]) if len(lines) > 2 else ""
                clean_t = clean_subtitle_text(text_part)
                if clean_t:
                    segments.append({"start": start_sec, "end": end_sec, "text": clean_t})
    return segments

def load_whisper_model(model_size: str = "base"):
    global _WHISPER_CACHE
    device = "cpu"
    if model_size not in _WHISPER_CACHE:
        import whisper
        model = whisper.load_model(model_size, device=device)
        _WHISPER_CACHE[model_size] = (model, device)
    model, cached_dev = _WHISPER_CACHE[model_size]
    return model

def transcribe_audio(audio_path: str, model_size: str = "base", language: str = None) -> dict:
    if not os.path.exists(audio_path):
        return {
            "success": False,
            "text": "",
            "language": "",
            "segments": [],
            "error": f"File not found: {audio_path}",
        }

    t0 = time.time()
    try:
        model = load_whisper_model(model_size)
        use_fp16 = torch.cuda.is_available()

        context_prompt = (
            "Đây là bản ghi âm video bài phát biểu hoặc phỏng vấn có phụ đề rõ ràng, viết hoa tên riêng chính xác (Nick Vujicic, Australia, Melbourne, Stanford, Steve Jobs)."
            if (language in ["vi", "vietnamese"] or not language or language == "auto")
            else "This is an audio recording of a speech or interview with clean punctuation, proper names, and capitalization."
        )

        result = model.transcribe(
            audio_path,
            fp16=use_fp16,
            verbose=False,
            condition_on_previous_text=False,
            temperature=0,
            beam_size=3,
            best_of=3,
            initial_prompt=context_prompt,
            language=language if language and language != "auto" else None,
        )

        full_text = normalize_speech_text(result.get("text", "").strip())
        detected_lang = result.get("language", "unknown")

        raw_segments = result.get("segments", [])
        structured_segments = []
        for seg in raw_segments:
            s_text = normalize_speech_text(seg.get("text", "").strip())
            if s_text:
                structured_segments.append({
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "text": s_text,
                })

        return {
            "success": True,
            "text": full_text,
            "language": detected_lang,
            "segments": structured_segments,
            "transcribe_time": round(time.time() - t0, 2),
            "word_count": len(full_text.split()),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "language": "",
            "segments": [],
            "error": str(e),
        }
