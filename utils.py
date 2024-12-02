from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
import json, os
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures

DATA_DIR = "./dataset/data"
VIDEO_DIR = "./dataset/videos"
BASE_URL = "https://www.youtube.com/watch?v="


def get_wholistic_df():
    df = pd.DataFrame(columns=["label", "youtube_id", "time_start", "time_end", "lang"])
    for csv_name in os.listdir(DATA_DIR):
        new_df = pd.read_csv(f"{DATA_DIR}/{csv_name}")
        df = pd.concat([df, new_df]).reset_index(drop=True)
    return df


def download_by_pandas_id(row, full_video=False):
    label, id, start, end, lang = row
    url = BASE_URL + id

    label = label.replace(" ", "_").lower()

    if not os.path.exists(f"{VIDEO_DIR}/{label}"):
        os.makedirs(f"{VIDEO_DIR}/{label}", exist_ok=True)

    if not full_video:
        out_path = os.path.join(
            "./dataset/videos",
            label,
            "%(id)s" + f"=s{start}e{end}" + "." + "%(ext)s",
        )
        options = {
            # can use 'title' or 'id'
            "outtmpl": out_path,
            "format": f"bestvideo[height<={720}]",
            "download_ranges": download_range_func(None, [(start, end)]),
            "force_keyframes_at_cuts": True,
            "quiet": True,
            "ignoreerrors": True,
            "cookies": "cookies.txt",
        }
    else:
        out_path = os.path.join(
            "./dataset/full_video_examples",
            label,
            "%(id)s.%(ext)s",
        )
        options = {
            # can use 'title' or 'id'
            "outtmpl": out_path,
            "format": f"bestvideo[height<={720}]",
            "quiet": True,
            "ignoreerrors": True,
            "cookies": "cookies.txt",
        }

    with YoutubeDL(options) as ydl:
        ydl.download([url])


def download_with_concurrency(df, full_video):
    with tqdm(total=len(df)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(download_by_pandas_id, row, full_video): id
                for id, row in df.iterrows()
            }
            for future in concurrent.futures.as_completed(future_to_url):
                pbar.update(1)


def download_all_subclip_videos():
    df = get_wholistic_df()
    df["label"] = (
        df["label"].apply(lambda x: x.replace(" ", "_")).apply(lambda x: x.lower())
    )
    with tqdm(total=len(df)) as pbar:

        for _, row in df.iterrows():
            label, id, start, end, lang = row
            url = BASE_URL + id

            if not os.path.exists(f"{VIDEO_DIR}/{label}"):
                os.makedirs(f"{VIDEO_DIR}/{label}", exist_ok=True)

            out_path = os.path.join(
                "./dataset/videos",
                label,
                "%(id)s" + f"=s{start}e{end}" + "." + "%(ext)s",
            )
            with YoutubeDL(
                {
                    # can use 'title' or 'id'
                    "outtmpl": out_path,
                    "format": f"bestvideo[height<={720}]",
                    "download_ranges": download_range_func(None, [(start, end)]),
                    "force_keyframes_at_cuts": True,
                    "quiet": True,
                    "ignoreerrors": True,
                    "cookies": "cookies.txt",
                }
            ) as ydl:
                ydl.download([url])
                pbar.update(1)


def download_videos_from_urls(URLS: list[str], desired_resolution="720"):
    # can use 'title' or 'id'
    ydl_opts = {
        "outtmpl": "./dataset/videos/%(id)s.%(ext)s",
        "format": f"bestvideo[height<={desired_resolution}]",
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(URLS)


def get_info_from_url(URL, desired_resolution="720"):
    ydl_opts = {"format": f"bestvideo[height<={desired_resolution}]"}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(URL, download=False)

        formats = info["formats"]
        for format in formats:
            print(
                f"Format ID: {format['format_id']} | Resolution: {format.get('height', 'N/A')} | Extension: {format['ext']} | URL: {format['url']}"
            )
        print(json.dumps(ydl.sanitize_info(info)))
