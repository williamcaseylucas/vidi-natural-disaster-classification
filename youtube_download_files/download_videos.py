# Uncomment either OPTION 2 function or OPTION 3 function

from youtube_download_files.utils import (
    download_with_concurrency,
    get_wholistic_df,
)
import random, pandas as pd


# OPTION 2 - FAST (for videos folder)
def download_for_videos_folder():
    df = get_wholistic_df()
    download_with_concurrency(df)


# download_for_videos_folder()


# OPTION 3 - 5 RANDOM FULL VIDEOS (for full_video_examples folder)
def download_full_videos_for_full_video_examples_folder():
    df = get_wholistic_df()
    data = []
    for _ in range(5):
        video_data = random.choice(df.values)
        data.append(video_data)

    new_df = pd.DataFrame(data, columns=df.columns)
    download_with_concurrency(new_df, full_video=True)


# download_with_concurrency(df, full_video=True)
