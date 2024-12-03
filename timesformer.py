"""

TO TRAIN A CLASSIFIER TO DETERMINE THE TYPE OF NATURAL DISASTER IN A VIDEO


"""

import numpy as np
import cv2, os
from PIL import Image
from torch import nn
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
)
from utils import get_wholistic_df
from torchinfo import summary

device = torch.device("mps")

np.random.seed(0)

image_processor = AutoImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics", device=device
)
model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
).to(device)


classes_len = len(os.listdir("./dataset/videos"))

for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(model.classifier.in_features, classes_len)

summary(
    model,
    input_size=(1, 8, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"],
)


def get_loader(
    df,
    transform,
    batch_size=32,
    num_workers=9,
    shuffle=True,
    pin_memory=True,
):

    dataset = VIDIDataset(
        df=df,
        transform=transform,
    )

    train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     shuffle=shuffle,
    #     pin_memory=pin_memory,
    #     collate_fn=MyColate(),
    # )
    # test_loader = DataLoader(
    #     dataset=test_set,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     shuffle=shuffle,
    #     pin_memory=pin_memory,
    #     collate_fn=MyColate(),
    # )

    return train_set, test_set


class MyColate:
    def __init__(self):
        pass

    def __call__(self, batch):
        video_frames = [item[0].unsqueeze(0) for item in batch]
        torch_video = torch.cat(video_frames, dim=0)
        labels = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)

        return {"pixel_values": torch_video, "labels": labels}
        # return torch_video, labels


class VIDIDataset(Dataset):

    def __init__(self, df, transform=None, sample_rate=8):
        self.transform = transform
        self.sample_rate = sample_rate
        partial_path = f"./dataset/videos"

        self.labels = os.listdir(partial_path)
        self.label_to_ids = {
            label: {
                path.split(".")[0].split("=")[0]: os.path.join(
                    partial_path, label, path
                )
                for path in os.listdir(f"./dataset/videos/{label}")
            }
            for label in self.labels
        }
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        indexes = df.apply(
            lambda x: x["youtube_id"]
            in self.label_to_ids[x["label"].replace(" ", "_").lower()],
            axis=1,
        )
        self.df = df[indexes].reset_index(
            drop=True
        )  # filter out the videos that are not in the dataset

        self.df_labels = self.df["label"]
        self.ids = self.df["youtube_id"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        df_label = self.df_labels[idx]

        mapped_label = df_label.replace(" ", "_").lower()
        if video_id not in self.label_to_ids[mapped_label]:
            return

        video_path = self.label_to_ids[mapped_label][video_id]
        video_frames = self.read_frames(video_path, self.sample_rate)

        if self.transform:
            video_frames = self.transform(list(video_frames), return_tensors="pt")[
                "pixel_values"
            ]

        return video_frames.squeeze(0), torch.tensor(self.label_to_idx[mapped_label])

    def read_frames(self, video_path, sample_rate):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            color_coverted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(color_coverted_frame)
        cap.release()
        frames = np.array(frames)
        indicies = np.linspace(0, len(frames) - 1, num=sample_rate).astype(np.int8)
        return [Image.fromarray(f) for f in frames[indicies]]


df = get_wholistic_df()
labels = os.listdir("./dataset/videos")
label_to_idx = {label: i for i, label in enumerate(labels)}
idx_to_label = {v: k for k, v in label_to_idx.items()}

train_set, test_set = get_loader(df=df, transform=image_processor)

training_args = TrainingArguments(
    output_dir="./timesformer_training/results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./timesformer_training/logs",
    learning_rate=1e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=MyColate(),
)

trainer.train()

model = TimesformerForVideoClassification.from_pretrained(
    "./timesformer_training/results/checkpoint-10"
).to(device)

model.eval()

video, label = test_set[0]
video.shape
idx_to_label[label.item()]
with torch.no_grad():
    res = model(**{"pixel_values": video.unsqueeze(0).to(device)})
    res.logits.shape

# -----------------

# from transformers import AutoImageProcessor, TimesformerForVideoClassification
# import numpy as np
# import torch

# video = list(np.ones((8, 3, 224, 224)))

# processor = AutoImageProcessor.from_pretrained(
#     "facebook/timesformer-base-finetuned-k400"
# )
# model = TimesformerForVideoClassification.from_pretrained(
#     "facebook/timesformer-base-finetuned-k400"
# )

# inputs = processor(video, return_tensors="pt")
# inputs.keys()
