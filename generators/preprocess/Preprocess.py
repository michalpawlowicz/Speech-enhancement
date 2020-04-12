import random
import os
import librosa
import numpy as np
from progress.bar import Bar
from typing import List
from generators.preprocess.Utils import read_files


def create(noise_dir: str, speech_dir: str, out_dir: str, frame_lenght: int, hop: int, sampling: int) -> None:
    """[summary]
    
    Arguments:
        noise_dir {str} -- [description]
        speech_dir {str} -- [description]
        out_dir {str} -- [description]
        frame_lenght {int} -- [description]
        hop {int} -- [description]
        sampling {int} -- [description]
    """
    print("-> Reading noise files")
    noice_files = list(os.scandir(noise_dir))

    print("-> Reading speech files")
    speech_files = list(os.scandir(speech_dir))

    print("-> Shuffling noise files")
    random.shuffle(noice_files)
    print("-> Shuffling speech files")
    random.shuffle(speech_files)

    print("-> Reading noise into memory")
    noise_frames = read_files(noice_files, sampling, frame_lenght, hop)

    bar = Bar('Blending', max=len(speech_files))
    for sample_file in speech_files:
        y, _ = librosa.load(sample_file, sr=sampling)
        y = np.vstack([y[s:s + frame_lenght]
                       for s in range(0, len(y) - frame_lenght + 1, frame_lenght)])
        indexes = np.random.randint(0, noise_frames.shape[0], y.shape[0])
        magnitutes = np.random.uniform(.2, .6, y.shape[0])
        for i, index in enumerate(indexes):
            y[i, :] = y[i, :] + magnitutes[i] * noise_frames[index, :]
        filename = os.path.splitext(os.path.basename(sample_file))[0] + ".wav"
        librosa.output.write_wav(os.path.join(
            out_dir, filename), y.reshape(1, -1)[0], sr=sampling)
        bar.next()
    bar.finish()
