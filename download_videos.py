from utils import (
    download_all_subclip_videos,
    download_with_concurrency,
    get_wholistic_df,
)
import random, pandas as pd

the_new_thing_dies_acoustic_url = "https://www.youtube.com/watch?v=9bsVvcPKzMM"

# OPTION 1 - SLOW
# download_all_subclip_videos()

# OPTION 2 - FAST
# df = get_wholistic_df()
# download_with_concurrency(df)

# OPTION 3 - RANDOM FULL VIDEOS
df = get_wholistic_df()
data = []
for _ in range(5):
    video_data = random.choice(df.values)
    data.append(video_data)

new_df = pd.DataFrame(data, columns=df.columns)
download_with_concurrency(new_df, full_video=True)
