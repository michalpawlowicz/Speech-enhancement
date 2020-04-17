from environment import check_environment_variables, variables
from preprocess import create, samplify, spectrogramplify, count_samples
from sklearn.preprocessing import normalize
import os
import librosa

env = check_environment_variables(variables)
frame_length = int(env["FRAME_LENGTH"])
hop = int(env["HOP"])
sampling = int(env["SAMPLING"])
samplify_npy_size = int(env["SAMPLIFY_NPY_SIZE"])
n_fft = int(env["N_FFT"])
fft_hop_length = int(env["STFT_HOP_LENGTH"])

create(env["INPUT_TRAIN_NOISE"], env["INPUT_TRAIN_CLEAN"],
       env["TRAIN_NOISY"], env["TRAIN_CLEAN"], frame_length, hop, sampling)

train_clean = [p.path for p in os.scandir(env["TRAIN_CLEAN"])]
train_clean = sorted(train_clean, key=lambda p: os.path.basename(p))

train_noisy = [p.path for p in os.scandir(env["TRAIN_NOISY"])]
train_noisy = sorted(train_noisy, key=lambda p: os.path.basename(p))

if len(train_clean) != len(train_noisy):
    raise RuntimeError("Different size!")

for x, y in zip(train_clean, train_noisy):
    if os.path.basename(x) != os.path.basename(y):
        raise RuntimeError(
            "Different sample sets! broke on {0} vs {1}".format(x, y))

samplify(train_clean, env["SAMPLIFY_TRAIN_CLEAN"],
         frame_length, hop, sampling, samplify_npy_size)
samplify(train_noisy, env["SAMPLIFY_TRAIN_NOISY"],
         frame_length, hop, sampling, samplify_npy_size)

samplifiy_train_clean = [
    p.path for p in os.scandir(env["SAMPLIFY_TRAIN_CLEAN"])]
samplifiy_train_clean = sorted(samplifiy_train_clean, key=lambda p: int(os.path.basename(p).split('.')[0]))

samplifiy_train_noisy = [
    p.path for p in os.scandir(env["SAMPLIFY_TRAIN_NOISY"])]
samplifiy_train_noisy = sorted(samplifiy_train_noisy, key=lambda p: int(os.path.basename(p).split('.')[0]))

spectrogramplify(samplifiy_train_clean,
                 env["SPECTROGRAM_TRAIN_CLEAN"], n_fft, fft_hop_length)
spectrogramplify(samplifiy_train_noisy,
                 env["SPECTROGRAM_TRAIN_NOISY"], n_fft, fft_hop_length)
