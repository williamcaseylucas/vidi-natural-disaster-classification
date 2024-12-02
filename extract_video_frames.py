import av
import numpy as np
from tqdm import tqdm


def get_frames_from_video(
    file_path="./dataset/full_video_examples/volcanic_eruption/VBTAcACmcgo.mp4",
    interval_of_window=10,
    frame_sample_rate=6,
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

        filtered_array = []
        for f in new_frames:
            size = f.shape[0]
            indices = np.linspace(0, size - 1, frame_sample_rate).astype(np.int16)
            filtered_array.append(f[indices])
        pbar.update(1)
        return np.stack([f for f in filtered_array]).transpose(
            0, 1, 4, 2, 3
        )  # batches, frames, height, width, channels
