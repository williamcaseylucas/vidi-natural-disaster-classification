# vidi-natural-disaster-classification

# Notes

- timesformer_training/results will be blank so will need to run timesformer.py to train model for checkpoint (too big to upload to github)
- dataset/videos will contain folders of the classification of the event
  - Will need to run youtube_download_files to get on your machine (too big to upload to youtube)
  - .mp4 and .webm format
- dataset/videos_processed will contain folders of the classification of the event but are in .pt format for faster loading
  - This can be achieved by looking at the timesformer.py file
- dataset/labels.json
  - contains number mappings to classification
- full_video_examples
  - Put data here that you want to test the model on for longer form videos
- For caption_models.py, interval changes the interval for the summartive video. 10 means 10 seconds worth of video is being crunched at a time. 5 means 5 seconds of video is being crunched at a time.

# Dataset

- https://vididataset.github.io/VIDI/
- pip install --upgrade youtube-dlp
- yt-dlp --cookies-from-browser chrome --cookies cookies.txt

# How to download youtube videos

# Scripts

- caption_models.py
  - contains utility functions to use GitCaptioner and Timesformer
- timesformer.py
  - contains code to train classifier on natural disaster video clips
- llama-3.py
  - main entrypoint where you chat with a model and get summary information about a video

# Folders

- dataset
  - data
    - contains csvs of links to youtube videos with labels
  - full_video_examples
    - contains a few full video samples to test
  - videos
    - raw videos
  - videos_processed
    - videos in .pt format for faster loading
  - labels.json
    - file that maps each label to a video classification
