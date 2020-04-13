export ROOT=./data

export TRAIN_CLEAN=/home/michal/OpenSLR/LibriSpeech/train-clean-100/
export TRAIN_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export TEST_CLEAN=/home/michal/OpenSLR/LibriSpeech/test-clean
export TEST_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export TRAIN_NOISY=$ROOT/Train/sound/
export TEST_NOISY=$ROOT/Test/sound/

export SAMPLING=16000
export FRAME_LENGTH=8192
export HOP=1024

# fft parameters
export STFT_HOP_LENGTH=64
export N_FFT=255

# directory to store model's checkpoint
export CHECKPOINTS_DIR=./checkpoints/

export BATCH_SIZE=128

export SAMPLIFY_TRAIN_CLEAN=$ROOT/Train/samplify/clean/
export SAMPLIFY_TRAIN_NOISY=$ROOT/Train/samplify/noisy/
export SAMPLIFY_TEST_CLEAN=$ROOT/Test/samplify/clean/
export SAMPLIFY_TEST_NOISY=$ROOT/Test/samplify/noisy/

# number of samples in npy dump
export SAMPLIFY_NPY_SIZE=100000

export SPECTROGRAM_TRAIN_CLEAN=$ROOT/Train/spectrogram/clean/
export SPECTROGRAM_TRAIN_NOISY=$ROOT/Train/spectrogram/noisy/
export SPECTROGRAM_TEST_CLEAN=$ROOT/Test/spectrogram/clean/
export SPECTROGRAM_TEST_NOISY=$ROOT/Test/spectrogram/noisy/

