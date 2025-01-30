from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

url = "https://www.youtube.com/watch?v="
id = "rBnJ7KB9rPc"
with YoutubeDL(
    {
        # can use 'title' or 'id'
        "outtmpl": "%(title)s.%(ext)s",
    }
) as ydl:
    ydl.download([url + id])
