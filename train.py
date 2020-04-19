import os
import numpy as np
import tensorflow as tf
import logging
import time
import keras
from environment import check_environment_variables, variables
from model import get_unet, unet
from typing import List

# Turn off tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def train_entry(**kwargs):
    pass

class Generator(tf.keras.utils.Sequence):
    def __init__(self, x_npy_files: List[str], y_npy_files: List[str], batch_size: int):
        self.batch_size = batch_size

        self.sample_count = 0
        for npy_file in x_npy_files:
            shape = np.load(npy_file).shape
            self.sample_count += shape[0]

        self.x_npy_files = x_npy_files
        self.y_npy_files = y_npy_files

        self.on_each_epoch()

    def __len__(self):
        return int(np.floor(self.sample_count / self.batch_size))

    def __getitem__(self, index):
        if index == 0:
            self.on_each_epoch()
        x_batch = []
        y_batch = []
        for _ in range(0, self.batch_size):
            x_batch.append(next(self.x_generator))
            y_batch.append(next(self.y_generator))
        return np.array(x_batch), np.array(y_batch)

    def on_each_epoch(self):
        self.x_samples_list = map(
            lambda npy_file: np.load(npy_file), self.x_npy_files)
        self.x_samples_list = map(lambda m: m.reshape(
            m.shape[0], m.shape[1], m.shape[2], 1), self.x_samples_list)

        self.y_samples_list = map(
            lambda npy_file: np.load(npy_file), self.y_npy_files)
        self.y_samples_list = map(lambda m: m.reshape(
            m.shape[0], m.shape[1], m.shape[2], 1), self.y_samples_list)

        def lazy_numpy_vstack(matrices):
            for matrix in matrices:
                yield from matrix

        self.x_generator = lazy_numpy_vstack(self.x_samples_list)
        self.y_generator = lazy_numpy_vstack(self.y_samples_list)



env = check_environment_variables(variables)

X_paths = sorted(map(lambda direntry: direntry.path,
                     os.scandir(env["SPECTROGRAM_TRAIN_NOISY"])))
y_paths = sorted(map(lambda direntry: direntry.path,
                     os.scandir(env["SPECTROGRAM_TRAIN_CLEAN"])))

g = Generator(X_paths, y_paths, 128)

X_test_paths = sorted(map(lambda direntry: direntry.path,
                     os.scandir(env["SPECTROGRAM_TEST_NOISY"])))
y_test_paths = sorted(map(lambda direntry: direntry.path,
                     os.scandir(env["SPECTROGRAM_TEST_CLEAN"])))

g_test = Generator(X_test_paths, y_test_paths, 128)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

# Save best models to disk during training
base_name = "data_nfft_{0}_fft_hop_{1}_frame_length_{2}".format(
    env["N_FFT"], env["STFT_HOP_LENGTH"], env["FRAME_LENGTH"])
checkpoint_name = base_name + "-cp-epoch_{epoch:04d}.h5"
checkpoint_path = os.path.join(
    "/home/michal/Documents/Speech-enhancement/checkpoints", checkpoint_name)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto', period=1)

log_dir = "logs/model-{0}-{1}".format(base_name, int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, update_freq='epoch', write_graph=True, profile_batch=0)

model = unet(input_size=(128, 128, 1))
model.fit_generator(g, epochs=50, shuffle=True, callbacks=[
                    checkpoint, tensorboard_callback], verbose=1, workers=6, use_multiprocessing=True, validation_data=g_test, validation_freq=1)
