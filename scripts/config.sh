export SAMPLING=16000
export FRAME_LENGTH=16384

# fft parameters
export STFT_HOP_LENGTH=128
export N_FFT=255

# directory to store model's checkpoint
export CHECKPOINTS_DIR=./checkpoints/

export BATCH_SIZE=64

export ROOT=./"data_nfft_"$N_FFT"_fft_hop_"$STFT_HOP_LENGTH"_frame_length_"$FRAME_LENGTH

export INPUT_TRAIN_CLEAN=/home/michal/OpenSLR/LibriSpeech/train-clean-100/
export INPUT_TRAIN_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export INPUT_TEST_CLEAN=/home/michal/OpenSLR/LibriSpeech/test-clean
export INPUT_TEST_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export TRAIN_NOISY=$ROOT/Train/noisy
export TEST_NOISY=$ROOT/Test/noisy

export TRAIN_CLEAN=$ROOT/Train/clean/
export TEST_CLEAN=$ROOT/Test/clean

export SAMPLIFY_TRAIN_CLEAN=$ROOT/Train/samplify/clean/
export SAMPLIFY_TRAIN_NOISY=$ROOT/Train/samplify/noisy/
export SAMPLIFY_TEST_CLEAN=$ROOT/Test/samplify/clean/
export SAMPLIFY_TEST_NOISY=$ROOT/Test/samplify/noisy/

# number of samples in npy dump
export SAMPLIFY_NPY_SIZE=2000

export SPECTROGRAM_TRAIN_CLEAN=$ROOT/Train/spectrogram/clean/
export SPECTROGRAM_TRAIN_NOISY=$ROOT/Train/spectrogram/noisy/
export SPECTROGRAM_TEST_CLEAN=$ROOT/Test/spectrogram/clean/
export SPECTROGRAM_TEST_NOISY=$ROOT/Test/spectrogram/noisy/


export MODEL=./checkpoints/best.h5
export PRED_IN_AUDIO=$ROOT/noisy.wav
export PRED_OUT_AUDIO=$ROOT/clean-predicted.wav
