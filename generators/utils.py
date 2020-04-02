import librosa
import numpy as np


def read_sample(sample_path: str, sampling: int, frame_lenght: int, hop: int):
    """[summary]

    Arguments:
        sample_path {str} -- [description]
        sampling {int} -- [description]
        frame_lenght {int} -- [description]
        hop {int} -- [description]

    Returns:
        [type] -- [description]
    """
    y, _ = librosa.load(sample_path, sr=sampling)
    return np.vstack([y[s:s + frame_lenght] for s in range(0, len(y) - frame_lenght + 1, hop)])
