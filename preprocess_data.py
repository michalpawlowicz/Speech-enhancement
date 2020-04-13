from environment import check_environment_variables, variables
from preprocess import create, samplify, spectrogramplify, count_samples
import os
import librosa

env = check_environment_variables(variables)
frame_length = int(env["FRAME_LENGTH"])
hop = int(env["HOP"])
sampling = int(env["SAMPLING"])
samplify_npy_size = int(env["SAMPLIFY_NPY_SIZE"])
n_fft = int(env["N_FFT"])
fft_hop_length = int(env["STFT_HOP_LENGTH"])

create(env["TRAIN_NOISE"], env["TRAIN_CLEAN"],
       env["TRAIN_NOISY"], frame_length, hop, sampling)

samples_nb = count_samples(
    list(os.scandir(env["TRAIN_NOISY"])), sampling, frame_length)

train_clean = [p.path for p in os.scandir(env["TRAIN_CLEAN"])]
sorted(train_clean, key=lambda p: os.path.basename(p))

train_noisy = [p.path for p in os.scandir(env["TRAIN_NOISY"])]
sorted(train_noisy, key=lambda p: os.path.basename(p))

if len(train_clean) != len(train_noisy):
    raise RuntimeError("Different size!")

for x, y in zip(train_clean, train_noisy):
    if os.path.basename(x) != os.path.basename(y):
        raise RuntimeError("Different sample sets! broke on {0} vs {1}".format(x, y))

samplify(train_clean, env["SAMPLIFY_TRAIN_CLEAN"],
         samples_nb, frame_length, hop, sampling, samplify_npy_size)
samplify(train_noisy, env["SAMPLIFY_TRAIN_NOISY"],
         samples_nb, frame_length, hop, sampling, samplify_npy_size)

samplifiy_train_clean = [
    p.path for p in os.scandir(env["SAMPLIFY_TRAIN_CLEAN"])]
samplifiy_train_noisy = [
    p.path for p in os.scandir(env["SAMPLIFY_TRAIN_NOISY"])]
spectrogramplify(samplifiy_train_clean,
                 env["SPECTROGRAM_TRAIN_CLEAN"], n_fft, fft_hop_length)
spectrogramplify(samplifiy_train_noisy,
                 env["SPECTROGRAM_TRAIN_NOISY"], n_fft, fft_hop_length)
