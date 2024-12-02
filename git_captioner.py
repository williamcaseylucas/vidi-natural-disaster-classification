from PIL import Image
import requests
from transformers import AutoProcessor, GitVisionModel
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm

from extract_video_frames import get_frames_from_video


def hidden_states_for_sample_image():

    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    model = GitVisionModel.from_pretrained("microsoft/git-base")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state


class GitCaptioner:
    def __init__(self):
        self.device = torch.device("mps")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/git-base-vatex", device=self.device
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-base-vatex"
        ).to(self.device)

        # set seed for reproducability
        np.random.seed(45)

    def get_extracted_frames(self, video_path):
        video_frames = get_frames_from_video(
            file_path=video_path,
            interval_of_window=10,
            frame_sample_rate=6,
        )

        video_frames.shape  # 29, 6, 3, 720, 1280

        batch_dim, frames = video_frames.shape[0:2]
        pixel_values = (
            self.processor(
                images=list(
                    video_frames.reshape(batch_dim * frames, *video_frames.shape[2:])
                ),
                return_tensors="pt",
            )
            .pixel_values[0]
            .to(self.device)
        )  # torch.Size([1, 6, 3, 224, 224])

        pixel_values = pixel_values.reshape(batch_dim, frames, *pixel_values.shape[1:])
        return pixel_values

    def get_tokens(self, pixel_values):
        # temporal embedding layer is of size 6 because we sample six frames and is (1 x 1 x 768)

        for tokens in pixel_values:
            # per token
            generated_ids = self.model.generate(
                pixel_values=tokens[None, :], max_length=50
            )  # 1, 11
            yield generated_ids

    def decode_caption(self, generated_ids):
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return caption

    def get_captions_and_intervals_in_seconds(
        self,
        video_path="./dataset/full_video_examples/volcanic_eruption/VBTAcACmcgo.mp4",
    ):
        frames = self.get_extracted_frames(video_path=video_path)
        captions = []
        start, end = 0, 10
        for token in tqdm(self.get_tokens(frames), total=len(frames), desc="Captions"):
            captions.append((start, end, self.decode_caption(token)))
            start += 10
            end += 10

        return captions
