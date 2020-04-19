from environment import check_environment_variables, variables
from preprocess import create, samplify, spectrogramplify, count_samples
from sklearn.preprocessing import normalize
import os
import librosa
from typing import Dict

env = check_environment_variables(variables)
frame_length = int(env["FRAME_LENGTH"])
sampling = int(env["SAMPLING"])
samplify_npy_size = int(env["SAMPLIFY_NPY_SIZE"])
n_fft = int(env["N_FFT"])
fft_hop_length = int(env["STFT_HOP_LENGTH"])


def preprocess_data_entry(**kwargs):
    pass
    """
    train_noisy = os.path.join(args["root_dir"], "Train", "noisy")
    test_noisy = os.path.join(args["root_dir"], "Test", "noisy")

    train_clean = os.path.join(args["root_dir"], "Train", "clean")
    test_clean = os.path.join(args["root_dir"], "Test", "clean")

    samplify_train_clean = os.path.join(
        args["root_dir"], "Train", "samplify", "clean")
    samplify_test_clean = os.path.join(
        args["root_dir"], "Test", "samplify", "clean")

    samplify_train_noisy = os.path.join(
        args["root_dir"], "Train", "samplify", "noisy")
    samplify_test_noisy = os.path.join(
        args["root_dir"], "Test", "samplify", "noisy")

    spectrogram_train_clean = os.path.join(
        args["root_dir"], "Train", "spectrogram", "clean")
    spectrogram_test_clean = os.path.join(
        args["root_dir"], "Test", "spectrogram", "clean")

    spectrogram_train_noisy = os.path.join(
        args["root_dir"], "Train", "spectrogram", "noisy")
    spectrogram_test_noisy = os.path.join(
        args["root_dir"], "Test", "spectrogram", "noisy")

    preprocess_data(env["input_train_noise"], env["input_train_clean"], train_noisy, train_clean, samplify_train_noisy,
                    samplify_train_clean, spectrogram_train_noisy, spectrogram_train_clean, args["frame_length"], args["sampling"])

    preprocess_data(env["input_test_noise"], env["input_test_clean"], test_noisy, test_clean, samplify_test_noisy,
                    samplify_test_clean, spectrogram_test_noisy, spectrogram_test_clean, args["frame_length"], args["sampling"])
    """


def preprocess_data(input_noise_dir: str, input_clean_dir: str, noisy_audio_dir: str, clean_audio_dir: str, samplify_noisy_dir: str, samplify_clean_dir: str, spectrogramify_clean_dir: str, spectrogramify_noisy_dir: str, frame_length: int, sampling: int):
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

    Raises:
        RuntimeError: [description]
        RuntimeError: [description]
    """

    create(input_noise_dir, input_clean_dir, noisy_audio_dir,
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


preprocess_data(env["INPUT_TRAIN_NOISE"], env["INPUT_TRAIN_CLEAN"], env["TRAIN_NOISY"], env["TRAIN_CLEAN"], env["SAMPLIFY_TRAIN_NOISY"],
                env["SAMPLIFY_TRAIN_CLEAN"], env["SPECTROGRAM_TRAIN_NOISY"], env["SPECTROGRAM_TRAIN_CLEAN"], frame_length, sampling)

preprocess_data(env["INPUT_TEST_NOISE"], env["INPUT_TEST_CLEAN"], env["TEST_NOISY"], env["TEST_CLEAN"], env["SAMPLIFY_TEST_NOISY"],
                env["SAMPLIFY_TEST_CLEAN"], env["SPECTROGRAM_TEST_NOISY"], env["SPECTROGRAM_TEST_CLEAN"], frame_length, sampling)
