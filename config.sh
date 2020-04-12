export TRAIN_CLEAN=/home/michal/OpenSLR/LibriSpeech/dev-clean
export TRAIN_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export TEST_CLEAN=/home/michal/OpenSLR/LibriSpeech/test-clean
export TEST_NOISE=/home/michal/ESC-50/ESC-50-master/audio

export TRAIN_NOISY=./data/Train/sound/
export TEST_NOISY=./data/Test/sound/

export SAMPLING=16000
export FRAME_LENGTH=8192
export HOP=1024

# fft parameters
export STFT_HOP_LENGTH=64
export N_FFT=255

# directory to store model's checkpoint
export CHECKPOINTS_DIR=./checkpoints/
