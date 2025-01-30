import av
import numpy as np
from tqdm import tqdm


def get_frames_from_video(
    file_path="./dataset/full_video_examples/volcanic_eruption/VBTAcACmcgo.mp4",
    interval_of_window=10,
    frame_sample_rate_for_caption=8,
    frame_sample_rate_for_classification=8,
):
    with tqdm(total=4) as pbar:
        container = av.open(file_path)
        stream = container.streams.video[0]
        fps = int(stream.average_rate)
        if stream.duration:
            float_time = float(stream.duration * stream.time_base)
            print(
                f"Video is {int(float_time / 60)} minutes and {int(float_time % 60)} seconds long."
            )

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        pbar.update(1)
        frames = np.stack([f for f in frames])
        pbar.update(1)

        chunk = fps * interval_of_window
        new_frames = np.array(
            [frames[i : i + chunk] for i in range(0, len(frames), chunk)], dtype=object
        )
        pbar.update(1)

        updated_frames_for_caption = []
        updated_frames_for_classification = []
        # frame rate may not be the same per window so we need to do this
        for f in new_frames:
            size = f.shape[0]
            caption_indices = np.linspace(
                0, size - 1, frame_sample_rate_for_caption
            ).astype(np.int16)
            classification_indices = np.linspace(
                0, size - 1, frame_sample_rate_for_classification
            ).astype(np.int16)
            updated_frames_for_caption.append(f[caption_indices])
            updated_frames_for_classification.append(f[classification_indices])

        pbar.update(1)
        return (
            np.stack([f for f in updated_frames_for_caption])
            .transpose(0, 1, 4, 2, 3)
            .astype(np.float32),
            np.stack([f for f in updated_frames_for_classification])
            .transpose(0, 1, 4, 2, 3)
            .astype(np.float32),
        )  # batches, frames, height, width, channels
