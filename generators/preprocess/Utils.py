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


def read_files(audio_files: List[str], sampling: int, frame_lenght: int, hop: int) -> np.ndarray:
    """[summary]

    Arguments:
        audio_files {List[str]} -- [description]
        sampling {int} -- [description]
        frame_lenght {int} -- [description]
        hop {int} -- [description]

    Returns:
        np.ndarray -- [description]
    """
    audio_stacks = []
    bar = Bar('Processing', max=len(audio_files))
    for f in audio_files:
        audio_stacks.append(read_sample(
            sample_path=f, sampling=sampling, frame_lenght=frame_lenght, hop=hop))
        bar.next()
    bar.finish()
    return np.vstack(audio_stacks)


def blend(speech_frames: np.ndarray, noise_frames: np.ndarray, frame_lenght: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[summary]

    Arguments:
        speech_frames {np.ndarray} -- [description]
        noise_frames {np.ndarray} -- [description]
        frame_lenght {int} -- [description]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray] -- [description]
    """

    noise = np.zeros((speech_frames.shape[0], frame_lenght))
    speech = np.zeros((speech_frames.shape[0], frame_lenght))
    noisy_speech = np.zeros((speech_frames.shape[0], frame_lenght))

    bar = Bar('Blending', max=speech_frames.shape[0])
    for i in range(speech_frames.shape[0]):
        noise_magnitute = np.random.uniform(0.2, 0.7)
        noise[i, :] = noise_magnitute * \
            noise_frames[np.random.randint(0, noise_frames.shape[0]), :]
        speech[i, :] = speech_frames[i, :]
        noisy_speech[i, :] = speech[i, :] + noise[i, :]
        bar.next()
    bar.finish()
    return noise, speech, noisy_speech


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
