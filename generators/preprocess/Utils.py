import numpy as np
import librosa
from progress.bar import Bar
from typing import List, Tuple


def read_sample(sample_path: str, sampling: int, frame_lenght: int, hop: int) -> np.ndarray:
    """[summary]

    Arguments:
        sample_path {str} -- [description]
        sampling {int} -- [description]
        frame_lenght {int} -- [description]
        hop {int} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    y, _ = librosa.load(sample_path, sr=sampling)
    return np.vstack([y[s:s + frame_lenght] for s in range(0, len(y) - frame_lenght + 1, hop)])


def read_files(audio_files: List[str], sampling: int, frame_length: int, hop: int) -> np.ndarray:
    """[summary]

    Arguments:
        audio_files {List[str]} -- [description]
        sampling {int} -- [description]
        frame_length {int} -- [description]
        hop {int} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    audio_stacks = []
    bar = Bar('Processing', max=len(audio_files))
    for f in audio_files:
        audio_stacks.append(read_sample(
            sample_path=f, sampling=sampling, frame_lenght=frame_length, hop=hop))
        bar.next()
    bar.finish()
    return np.vstack(audio_stacks)


def count_samples(audio_files: List[str], sampling: int, frame_lenght: int, hop: int) -> int:
    """[summary]

    Arguments:
        audio_files {List[str]} -- [description]
        sampling {int} -- [description]
        frame_lenght {int} -- [description]
        hop {int} -- [description]

    Returns:
        int -- [description]
    """
    count = 0
    for f in audio_files:
        count += read_sample(f, sampling, frame_lenght, hop).shape[0]
    return count
