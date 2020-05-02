import random
import os
import librosa
import numpy as np
from progress.bar import Bar
from typing import List
from numpy import save
import math
from sklearn.preprocessing import MinMaxScaler
import pickle


def read_files(audio_files: List[str], sampling: int, frame_length: int, hop_length: int) -> np.ndarray:
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
            y, frame_length=frame_length, hop_length=hop_length, axis=0))
        bar.next()
    bar.finish()
    return np.vstack(audio_stacks)


def create(noise_dir: str, speech_dir: str, noisy_dir: str, clean_dir: str, frame_length: int, sampling: int, speech_frame_hop: float = 0.2, noise_frame_hop: float = 0.2) -> int:
    """[summary]

    Arguments:
        noise_dir {str} -- [description]
        speech_dir {str} -- [description]
        noisy_dir {str} -- [description]
        clean_dir {str} -- [description]
        frame_length {int} -- [description]
        sampling {int} -- [description]

    Returns:
        int -- [Total number of generated samples]
    """
    noice_files = list(os.scandir(noise_dir))
    speech_files = list(os.scandir(speech_dir))

    random.shuffle(noice_files)
    random.shuffle(speech_files)

    print("Reading noise into memory")
    noise_frames = read_files(noice_files, sampling,
                              frame_length, int(frame_length * speech_frame_hop))

    samples_count = 0

    bar = Bar('Blending samples', max=len(speech_files))
    for sample_file in speech_files:
        y, _ = librosa.load(sample_file, sr=sampling)
        y = librosa.util.frame(y, frame_length=frame_length,
                               hop_length=int(frame_length * noise_frame_hop), axis=0)
        samples_count += y.shape[0]
        filename = os.path.splitext(os.path.basename(sample_file))[0] + ".wav"
        librosa.output.write_wav(os.path.join(
            clean_dir, filename), y.reshape(1, -1)[0], sr=sampling)
        indexes = np.random.randint(0, noise_frames.shape[0], y.shape[0])
        magnitutes = np.random.uniform(.2, .6, y.shape[0])
        for i, index in enumerate(indexes):
            y[i, :] = y[i, :] + magnitutes[i] * noise_frames[index, :]
        librosa.output.write_wav(os.path.join(
            noisy_dir, filename), y.reshape(1, -1)[0], sr=sampling)
        bar.next()
    bar.finish()

    return samples_count


def samplify(audio_files: List[str], output_path: str, frame_length: int, sampling: int, npy_samples_count: int):
    """[summary]

    Arguments:
        audio_files {List[str]} -- [description]
        output_path {str} -- [description]
        frame_length {int} -- [description]
        sampling {int} -- [description]
        npy_samples_count {int} -- [description]
    """
    npy_frames = []
    npy_idx = 0
    samples_in_npy_frames = 0
    bar = Bar('Samplifying ..', max=len(audio_files))
    for audio_file in audio_files:
        y, _ = librosa.load(audio_file, sr=sampling)
        frames = librosa.util.frame(
            y, frame_length=frame_length, hop_length=int(frame_length * 0.3), axis=0)
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
    """[summary]

    Arguments:
        samples_npy {List[str]} -- [description]
        spectrogram_out {str} -- [description]
        n_fft {int} -- [description]
        fft_hop_length {int} -- [description]
    """
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

        spectrograms = np.array(spectrograms)

        output = os.path.join(spectrogram_out, "{}.npy".format(idx))
        print("\nWriting spectrograms to %s" % output)
        save(output, np.array(spectrograms))
        bar.finish()

def shuffle(x_samples_npy: List[str], y_samples_npy: List[str]):
    print("Shuffling denerated npy files")
    for x_npy, y_npy in zip(x_samples_npy, y_samples_npy):
        x = np.load(x_npy)
        y = np.load(y_npy)
        if len(x) != len(y):
            raise RuntimeError("Samle sets are of different sizes")
        indexes = list(range(0, len(x)))
        random.shuffle(indexes)
        for i, j in enumerate(indexes):
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]
        save(x_npy, x)
        save(y_npy, y)

def fit_scaler(samples_npy: List[str], scaler_save_path: str):
    scaler = MinMaxScaler()
    bar = Bar('Fitting scaler ..', max=len(samples_npy))
    for samples_path in samples_npy:
        samples = np.load(samples_path)
        shape = samples.shape
        scaler.partial_fit(samples.reshape(shape[0], -1))
        bar.next()
    bar.finish()
    print("Saving scaler to %s" % scaler_save_path)
    with open(scaler_save_path, 'wb+') as f:
        pickle.dump(scaler, f)


def scale_it(samples_npy: List[str], scaler_path: str):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    bar = Bar('Scaling ..', max=len(samples_npy))
    for samples_path in samples_npy:
        samples = np.load(samples_path)
        shape = samples.shape
        samples = samples.reshape(shape[0], -1)
        samples = scaler.transform(samples).reshape(shape)
        temp_output_file = samples_path.split('.')[0] + "_tmp.npy"
        save(temp_output_file, samples)
        os.rename(temp_output_file, samples_path)
        bar.next()
    bar.finish()
