from preprocess import create, samplify, spectrogramplify
from sklearn.preprocessing import normalize
from typing import Tuple
import os
import librosa


def preprocess_data_entry(**kwargs):
    sampling = kwargs["sampling"]
    frame_length = kwargs["frame_length"]
    workdir = kwargs["workdir"]
    n_fft = kwargs["n_fft"]
    fft_hop_length = kwargs["fft_hop_length"]
    samplify_npy_size = kwargs["samplify_npy_size"]

    train_noisy = os.path.join(workdir, "Train", "noisy")
    test_noisy = os.path.join(workdir, "Test", "noisy")

    train_clean = os.path.join(workdir, "Train", "clean")
    test_clean = os.path.join(workdir, "Test", "clean")

    samplify_train_clean = os.path.join(
        workdir, "Train", "samplify", "clean")
    samplify_test_clean = os.path.join(
        workdir, "Test", "samplify", "clean")

    samplify_train_noisy = os.path.join(
        workdir, "Train", "samplify", "noisy")
    samplify_test_noisy = os.path.join(
        workdir, "Test", "samplify", "noisy")

    spectrogram_train_clean = os.path.join(
        workdir, "Train", "spectrogram", "clean")
    spectrogram_test_clean = os.path.join(
        workdir, "Test", "spectrogram", "clean")

    spectrogram_train_noisy = os.path.join(
        workdir, "Train", "spectrogram", "noisy")
    spectrogram_test_noisy = os.path.join(
        workdir, "Test", "spectrogram", "noisy")

    train_samples_count = preprocess_data(kwargs["train"]["input_noise"], kwargs["train"]["input_clean"], train_noisy, train_clean, samplify_train_noisy,
                                          samplify_train_clean, spectrogram_train_noisy, spectrogram_train_clean, frame_length, sampling, samplify_npy_size, n_fft, fft_hop_length)

    test_samples_count = preprocess_data(kwargs["test"]["input_noise"], kwargs["test"]["input_clean"], test_noisy, test_clean, samplify_test_noisy,
                                         samplify_test_clean, spectrogram_test_noisy, spectrogram_test_clean, frame_length, sampling, samplify_npy_size, n_fft, fft_hop_length)

    return train_samples_count, test_samples_count


def preprocess_data(input_noise_dir: str, input_clean_dir: str,
                    noisy_audio_dir: str, clean_audio_dir: str,
                    samplify_noisy_dir: str, samplify_clean_dir: str,
                    spectrogramify_clean_dir: str, spectrogramify_noisy_dir: str,
                    frame_length: int, sampling: int, samplify_npy_size: int,
                    n_fft: int, fft_hop_length: int) -> int:
    """[summary]

    Arguments:
        input_noise_dir {str} -- [description]
        input_clean_dir {str} -- [description]
        noisy_audio_dir {str} -- [description]
        clean_audio_dir {str} -- [description]
        samplify_noisy_dir {str} -- [description]
        samplify_clean_dir {str} -- [description]
        spectrogramify_clean_dir {str} -- [description]
        spectrogramify_noisy_dir {str} -- [description]
        frame_length {int} -- [description]
        sampling {int} -- [description]
        samplify_npy_size {int} -- [description]
        n_fft {int} -- [description]
        fft_hop_length {int} -- [description]

    Raises:
        RuntimeError: [description]
        RuntimeError: [description]

    Returns:
        int -- [Total number of samples]
    """

    samples_count = create(input_noise_dir, input_clean_dir, noisy_audio_dir,
                           clean_audio_dir, frame_length, sampling)

    clean_audio_paths = sorted(
        map(lambda entry: entry.path, os.scandir(clean_audio_dir)))
    noisy_audio_paths = sorted(
        map(lambda entry: entry.path, os.scandir(noisy_audio_dir)))

    if len(clean_audio_paths) != len(noisy_audio_paths):
        raise RuntimeError("Different sets size!")

    if any(os.path.basename(x) != os.path.basename(y) for (x, y) in zip(clean_audio_paths, noisy_audio_paths)):
        raise RuntimeError("Different sample sets!")

    samplify(clean_audio_paths, samplify_clean_dir,
             frame_length, sampling, samplify_npy_size)
    samplify(noisy_audio_paths, samplify_noisy_dir,
             frame_length, sampling, samplify_npy_size)

    samplifiy_clean = sorted(
        map(lambda entry: entry.path, os.scandir(samplify_clean_dir)))
    samplifiy_noisy = sorted(
        map(lambda entry: entry.path, os.scandir(samplify_noisy_dir)))

    spectrogramplify(samplifiy_clean,
                     spectrogramify_clean_dir, n_fft, fft_hop_length)
    spectrogramplify(samplifiy_noisy,
                     spectrogramify_noisy_dir, n_fft, fft_hop_length)

    return samples_count
