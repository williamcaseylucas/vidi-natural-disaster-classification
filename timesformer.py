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
from utils import get_wholistic_df, write_frames
from torchinfo import summary
import evaluate

device = torch.device("mps")
classes_len = len(os.listdir("./dataset/videos"))

np.random.seed(0)

image_processor = AutoImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base-finetuned-kinetics", device=device
)
model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400",
).to(device)


# write_frames(root_dir="./dataset/videos", transform=image_processor, sample_rate=8)


for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Linear(model.classifier.in_features, classes_len)
model.num_labels = classes_len

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
        new_path = f"./dataset/videos_processed"

        self.labels = os.listdir(partial_path)
        self.label_to_ids = {
            label: {
                path.split(".")[0]: os.path.join(new_path, label, path)
                for path in os.listdir(f"./dataset/videos_processed/{label}")
            }
            for label in self.labels
        }
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        indexes = df.apply(
            lambda x: f'{x["youtube_id"]}=s{x['time_start']}e{x['time_end']}'
            in self.label_to_ids[x["label"].replace(" ", "_").lower()],
            axis=1,
        )
        self.df = df[indexes].reset_index(
            drop=True
        )  # filter out the videos that are not in the dataset

        self.df_labels = self.df["label"]
        self.ids = self.df["youtube_id"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        df_row = self.df.iloc[idx]
        start = df_row["time_start"]
        end = df_row["time_end"]
        id = f"{video_id}=s{start}e{end}"

        df_label = self.df_labels[idx]
        mapped_label = df_label.replace(" ", "_").lower()

        video_path = self.label_to_ids[mapped_label][id]

        video_frames = torch.load(video_path)

        return (
            # video_frames.squeeze(0),
            video_frames,
            torch.tensor(self.label_to_idx[mapped_label]),
            video_path.split("/")[-1],
        )


df = get_wholistic_df()
labels = os.listdir("./dataset/videos")
label_to_idx = {label: i for i, label in enumerate(labels)}
idx_to_label = {v: k for k, v in label_to_idx.items()}

train_set, test_set = get_loader(df=df, transform=image_processor)
metric = evaluate.load("accuracy")


# steps reported = batch_size * num_epochs
training_args = TrainingArguments(
    output_dir="./timesformer_training/results",
    num_train_epochs=25,
    per_device_train_batch_size=35,
    per_device_eval_batch_size=35,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./timesformer_training/logs",
    learning_rate=1e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


model = TimesformerForVideoClassification.from_pretrained(
    f"./timesformer_training/results/{os.listdir('./timesformer_training/results')[-1]}",
    num_labels=classes_len,
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=MyColate(),
    compute_metrics=compute_metrics,
)

# trainer.train(
#     resume_from_checkpoint=f"./timesformer_training/results/{os.listdir('./timesformer_training/results')[-1]}",
# )
# trainer.train()

# Best accuracy (ish) -> 'eval_accuracy': 0.855807743658211,

new_model = TimesformerForVideoClassification.from_pretrained(
    f"./timesformer_training/results/{os.listdir('./timesformer_training/results')[-1]}",
    num_labels=classes_len,
).to(device)

new_model.eval()

len(train_set)
len(test_set)
for video, label, video_id in test_set:
    english_label = idx_to_label[label.item()]
    with torch.no_grad():
        res = model(**{"pixel_values": video.unsqueeze(0).to(device)})
        class_label = res.logits.argmax(-1)
        print("file_path: ", f"./dataset/videos/{english_label}/{video_id}")
        print("actual label: ", english_label)
        print("predicted label: ", idx_to_label[class_label.item()])
        print("")

video.shape
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
