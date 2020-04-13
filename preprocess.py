import random
import os
import librosa
import numpy as np
from progress.bar import Bar
from typing import List
from numpy import save
import math


def read_files(audio_files: List[str], sampling: int, frame_length: int) -> np.ndarray:
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
        y, _ = librosa.load(f, sr=sampling)
        audio_stacks.append(librosa.util.frame(
            y, frame_length=frame_length, hop_length=frame_length, axis=0))
        bar.next()
    bar.finish()
    return np.vstack(audio_stacks)


def count_samples(audio_files: List[str], sampling: int, frame_length: int) -> int:
    """[summary]

    Arguments:
        audio_files {List[str]} -- [description]
        sampling {int} -- [description]
        frame_length {int} -- [description]

    Returns:
        int -- [description]
    """
    count = 0
    for f in audio_files:
        y, _ = librosa.load(f, sr=sampling)
        count += len(librosa.util.frame(y, frame_length=frame_length,
                                        hop_length=frame_length, axis=0))
    return count


def create(noise_dir: str, speech_dir: str, noisy_dir: str, frame_length: int, hop: int, sampling: int) -> None:
    """[summary]

    Arguments:
        noise_dir {str} -- [description]
        speech_dir {str} -- [description]
        noisy_dir {str} -- [description]
        frame_length {int} -- [description]
        hop {int} -- [description]
        sampling {int} -- [description]
    """
    noice_files = list(os.scandir(noise_dir))
    speech_files = list(os.scandir(speech_dir))

    random.shuffle(noice_files)
    random.shuffle(speech_files)

    print("Reading noise into memory")
    noise_frames = read_files(noice_files, sampling, frame_length)

    bar = Bar('Blending samples', max=len(speech_files))
    for sample_file in speech_files:
        y, _ = librosa.load(sample_file, sr=sampling)
        y = librosa.util.frame(y, frame_length=frame_length,
                               hop_length=frame_length, axis=0)
        indexes = np.random.randint(0, noise_frames.shape[0], y.shape[0])
        magnitutes = np.random.uniform(.2, .6, y.shape[0])
        for i, index in enumerate(indexes):
            y[i, :] = y[i, :] + magnitutes[i] * noise_frames[index, :]
        filename = os.path.splitext(os.path.basename(sample_file))[0] + ".wav"
        librosa.output.write_wav(os.path.join(
            noisy_dir, filename), y.reshape(1, -1)[0], sr=sampling)
        bar.next()
    bar.finish()


def samplify(audio_files: List[str], output_path: str, samples_nb: int, frame_length: int, hop: int, sampling: int, npy_samples_count: int):
    npy_frames = []
    npy_idx = 0
    samples_in_npy_frames = 0
    bar = Bar('Samplifying ..', max=len(audio_files))
    for audio_file in audio_files:
        y, _ = librosa.load(audio_file, sr=sampling)
        frames = librosa.util.frame(
            y, frame_length=frame_length, hop_length=frame_length, axis=0)
        if frames[-1].shape[0] < frame_length:
            frames = frames[0:-1]
        samples_in_npy_frames += frames.shape[0]
        npy_frames.append(frames)
        if samples_in_npy_frames > npy_samples_count:
            output = os.path.join(output_path, "{}.npy".format(npy_idx))
            v = np.vstack(npy_frames)
            print("\nWriting {0} into {1}".format(v.shape, output))
            save(output, v)
            npy_idx += 1
            npy_frames = []
            samples_in_npy_frames = 0
        bar.next()
    if len(npy_frames) > 0:
        output = os.path.join(output_path, "{}.npy".format(npy_idx))
        v = np.vstack(npy_frames)
        print("\nWriting {0} into {1}".format(v.shape, output))
        save(output, v)
    bar.finish()


def spectrogramplify(samples_npy: List[str], spectrogram_out: str, n_fft: int, fft_hop_length: int):
    for idx, samples_path in enumerate(samples_npy):
        print("\nProcessing %s" % samples_path)
        samples = np.load(samples_path)
        spectrograms = []
        bar = Bar('Spectrogramplify ..', max=len(samples))
        for sample in samples:
            magnitude, _ = librosa.magphase(librosa.stft(
                sample, n_fft=n_fft, hop_length=fft_hop_length))
            spectrograms.append(librosa.amplitude_to_db(magnitude, ref=np.max))
            bar.next()
        output = os.path.join(spectrogram_out, "{}.npy".format(idx))
        print("\nWriting spectrograms to %s" % output)
        save(output, np.array(spectrograms))
        bar.finish()
