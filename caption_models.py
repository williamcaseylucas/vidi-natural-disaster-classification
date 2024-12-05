import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
)
import torch
from tqdm import tqdm
from extract_video_frames import get_frames_from_video
from abc import ABC

from enum import Enum


import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification, logging
import json


class VideoCaptionType(Enum):
    # .name = .value
    GIT = "git"
    TIMESFORMER = "timesformer"


device = torch.device("mps")

logging.set_verbosity_error()


class TimesFormerClassifier:
    def __init__(self, pretrained_model_dir, device):
        # Don't need to use this since frames will already be processed
        self.image_processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics", device=device
        )

        # VIDI dataset has 43 classifications
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model_dir,
            num_labels=43,
        ).to(device)

        with open("./dataset/labels.json", "r") as f:
            labels = json.load(f)
            self.labels = list(labels["label"].values())

        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def get_predictions(self, frames):
        logits = self.model(frames).logits
        predictions = logits.argmax(-1)

        return [self.idx_to_label[pred.item()] for pred in predictions]


class Captioner(ABC):
    def __init__(self, with_classification=False):
        self.device = torch.device("mps")
        self.tokenizer = None
        self.classifier = None
        self.with_classification = with_classification

        if self.with_classification:
            self.classifier = TimesFormerClassifier(
                pretrained_model_dir="./timesformer_training/results/checkpoint-946",
                device=self.device,
            )

    def get_extracted_frames(self, video_path, interval_of_window=10):
        video_frames = get_frames_from_video(
            file_path=video_path,
            interval_of_window=interval_of_window,
            frame_sample_rate=self.clip_len,
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

    def get_tokens(
        self,
        pixel_values,
        gen_kwargs={
            "min_length": 10,
            "max_length": 50,
            # "num_beams": 8,
        },
    ):
        # temporal embedding layer is of size 6 because we sample six frames and is (1 x 1 x 768)
        for tokens in pixel_values:
            # per token
            generated_ids = self.model.generate(
                pixel_values=tokens[None, :], **gen_kwargs
            )  # 1, 11
            yield generated_ids

    def decode_caption(self, generated_ids):
        if self.tokenizer:
            caption = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

        return caption

    def get_captions_and_intervals_in_seconds(
        self,
        video_path="./dataset/full_video_examples/volcanic_eruption/VBTAcACmcgo.mp4",
        interval_of_window=10,
    ):
        frames = self.get_extracted_frames(
            video_path=video_path, interval_of_window=interval_of_window
        )
        classifications = None
        captions = []
        start, end = 0, interval_of_window
        for idx, token in enumerate(
            tqdm(self.get_tokens(frames), total=len(frames), desc="Captions")
        ):
            if self.with_classification:
                classifications = self.classifier.get_predictions(frames)
            captions.append(
                {
                    "start": start,
                    "end": end,
                    "caption": self.decode_caption(token),
                    "classification": classifications[idx],
                }
            )
            start += interval_of_window
            end += interval_of_window

        return captions


class GitCaptioner(Captioner):
    def __init__(self, with_classification):
        super().__init__(with_classification=with_classification)

        self.processor = AutoProcessor.from_pretrained(
            "microsoft/git-base-vatex", device=self.device
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/git-base-vatex"
        ).to(self.device)

        self.clip_len = self.model.config.num_image_with_embedding

        # set seed for reproducability
        np.random.seed(45)


class TimesformerCaptioner(Captioner):

    def __init__(self, with_classification):
        super().__init__(with_classification=with_classification)
        # load pretrained processor, tokenizer, and model
        self.processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base", device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", device=self.device)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "Neleac/timesformer-gpt2-video-captioning"
        ).to(self.device)

        self.clip_len = self.model.config.encoder.num_frames

        # set seed for reproducability
        np.random.seed(45)
