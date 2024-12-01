from PIL import Image
import requests
from transformers import AutoProcessor, GitVisionModel
import av
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForCausalLM


def hidden_states_for_sample_image():

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = GitVisionModel.from_pretrained("microsoft/git-base")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state


processor = AutoProcessor.from_pretrained("microsoft/git-base-vatex")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vatex")

# set seed for reproducability
np.random.seed(45)


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# load video
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
container = av.open(file_path)

# sample frames
num_frames = model.config.num_image_with_embedding
indices = sample_frame_indices(
    clip_len=num_frames, frame_sample_rate=4, seg_len=container.streams.video[0].frames
)
frames = read_video_pyav(
    container, indices
)  # 6, 360, 640, 3 -> frames,  height, width, channels

# path = "./dataset/videos/airplane_accident/3ioC2pswXPA=s329e338.mp4"
path = "./dataset/videos/airplane_accident/i2ikTy5wLSI=s292e297.mp4"
# path = "./dataset/videos/bus_accident/4pmotmnr99Y=s62e66.mp4"
# path = "./dataset/videos/bus_accident/HE_sybHJaQY=s88e96.mp4"
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

np_array = np.array(frames)
indices = np.linspace(0, len(frames) - 1, num=num_frames).astype(np.int16)
frames = np_array[indices]  # 6, 480, 640, 3

frames.shape  # 6, 360, 640, 3
pixel_values = processor(
    images=list(frames), return_tensors="pt"
).pixel_values  # torch.Size([1, 6, 3, 224, 224])

# temporal embedding layer is of size 6 because we sample six frames and is (1 x 1 x 768)
Image.fromarray(frames[3])
generated_ids = model.generate(pixel_values=pixel_values, max_length=50)  # 1, 11

print(
    "Generated caption:",
    processor.batch_decode(generated_ids, skip_special_tokens=True),
)


import os

captions = os.listdir("./dataset/videos")
