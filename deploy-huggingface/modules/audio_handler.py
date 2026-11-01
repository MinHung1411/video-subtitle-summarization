"""
Audio Ingestion Module:
Supports direct YouTube / Web URL downloading via yt-dlp
and local video / audio file handling.
"""

import os
import re
import json
import tempfile
import uuid

_DEMO_CAPTIONS_CACHE = {}
_DEMO_PATH = os.path.join(os.path.dirname(__file__), "demo_captions.json")
if os.path.exists(_DEMO_PATH):
    try:
        with open(_DEMO_PATH, "r", encoding="utf-8") as f:
            _DEMO_CAPTIONS_CACHE = json.load(f)
    except Exception:
        pass

def extract_youtube_video_id(url: str) -> str:
    if not url:
        return None
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/|v\/|shorts\/)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def fetch_youtube_captions(video_id: str) -> dict:
    if video_id in _DEMO_CAPTIONS_CACHE:
        cached = _DEMO_CAPTIONS_CACHE[video_id]
        return {
            "success": True,
            "segments": cached.get("segments", []),
            "text": cached.get("text", ""),
            "language": cached.get("language", "vi" if video_id == "VAIe6L68M3E" else "en"),
            "duration": cached.get("duration", 0.0),
            "title": cached.get("title", f"YouTube Video ({video_id})")
        }

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_data = None
        lang = "en"

        # Strategy 1: Prioritize list & find transcript (v1.x instance API)
        try:
            api = YouTubeTranscriptApi()
            if hasattr(api, 'list'):
                tl = api.list(video_id)
                t = None
                try:
                    t = tl.find_transcript(['vi', 'en', 'fr', 'es'])
                except Exception:
                    try:
                        t = tl.find_generated_transcript(['vi', 'en', 'fr', 'es'])
                    except Exception:
                        try:
                            t = next(iter(tl))
                        except Exception:
                            t = None
                if t:
                    transcript_data = t.fetch()
                    lang = getattr(t, 'language_code', lang)
        except Exception:
            pass

        # Strategy 2: Direct instance fetch
        if transcript_data is None:
            try:
                api = YouTubeTranscriptApi()
                if hasattr(api, 'fetch'):
                    transcript_data = api.fetch(video_id)
            except Exception:
                pass

        # Strategy 3: v0.x classmethod fallback
        if transcript_data is None and hasattr(YouTubeTranscriptApi, 'get_transcript'):
            try:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi', 'en', 'fr', 'es'])
            except Exception:
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                except Exception:
                    pass

        if not transcript_data:
            return {"success": False, "error": "No captions available for this video."}

        segments = []
        for it in transcript_data:
            txt = getattr(it, 'text', None) or (it.get('text') if isinstance(it, dict) else str(it))
            st = getattr(it, 'start', 0.0) or (it.get('start', 0.0) if isinstance(it, dict) else 0.0)
            dur = getattr(it, 'duration', 0.0) or (it.get('duration', 0.0) if isinstance(it, dict) else 0.0)
            if txt and txt.strip():
                clean_t = re.sub(r'[\r\n]+', ' ', txt).strip()
                segments.append({
                    'start': float(st),
                    'end': float(st) + float(dur),
                    'text': clean_t
                })

        full_text = " ".join([s['text'] for s in segments])
        total_duration = segments[-1]['end'] if segments else 0.0
        return {
            "success": True,
            "segments": segments,
            "text": full_text,
            "language": lang,
            "duration": total_duration,
            "title": f"YouTube Video ({video_id})"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def is_valid_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    regex = re.compile(
        r"^(?:http|ftp)s?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url.strip()) is not None

def download_audio_from_url(
    url: str, output_dir: str = None, max_duration_sec: int = 7200
) -> dict:
    try:
        import yt_dlp
    except ImportError:
        return {
            "success": False,
            "audio_path": None,
            "title": None,
            "duration": 0,
            "error": "Missing dependency: yt-dlp. Install via: pip install yt-dlp",
        }

    if not is_valid_url(url):
        return {
            "success": False,
            "audio_path": None,
            "title": None,
            "duration": 0,
            "error": "Invalid URL provided.",
        }

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    unique_id = uuid.uuid4().hex[:8]
    output_template = os.path.join(output_dir, f"audio_{unique_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Video")
            duration = info.get("duration", 0)

            if duration and duration > max_duration_sec:
                return {
                    "success": False,
                    "audio_path": None,
                    "title": title,
                    "duration": duration,
                    "error": f"Video exceeds max duration limit ({max_duration_sec // 60} mins).",
                }

            ydl.download([url])

            expected_audio = os.path.join(output_dir, f"audio_{unique_id}.mp3")
            if os.path.exists(expected_audio):
                return {
                    "success": True,
                    "audio_path": expected_audio,
                    "title": title,
                    "duration": duration,
                    "error": None,
                }
            return {
                "success": False,
                "audio_path": None,
                "title": title,
                "duration": duration,
                "error": "Extracted audio file not found.",
            }
    except Exception as e:
        return {
            "success": False,
            "audio_path": None,
            "title": None,
            "duration": 0,
            "error": str(e),
        }

def format_seconds_to_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hrs = seconds // 3600
    mins = (seconds % 3600) // 60
    secs = seconds % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"
