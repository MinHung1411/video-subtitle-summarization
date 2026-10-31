"""
Audio Ingestion Module:
Supports direct YouTube / Web URL downloading via yt-dlp
and local video / audio file handling.
"""

import os
import re
import tempfile
import uuid

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
