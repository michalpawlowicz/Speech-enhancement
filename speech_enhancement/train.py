import os
import numpy as np
import tensorflow as tf
import time
from speech_enhancement.model import get_unet, unet
from typing import List

# Turn off tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_entry(**kwargs):
    workdir = kwargs["workdir"]
    log_dir = kwargs["logs"]
    checkpoint_dir = kwargs["checkpoints"]
    epochs = kwargs["epochs"]
    batch_size = kwargs["batch_size"]
    input_size = kwargs["input_size"]
    validate = kwargs["validate"]
    loss = kwargs["loss"]
    optimizer = kwargs["optimizer"]

    spectrogram_train_clean = os.path.join(
        workdir, "Train", "spectrogram", "clean")
    spectrogram_test_clean = os.path.join(
        workdir, "Test", "spectrogram", "clean")

    spectrogram_train_noisy = os.path.join(
        workdir, "Train", "spectrogram", "noisy")
    spectrogram_test_noisy = os.path.join(
        workdir, "Test", "spectrogram", "noisy")

    X_paths = sorted(map(lambda direntry: direntry.path,
                         os.scandir(spectrogram_train_noisy)))
    y_paths = sorted(map(lambda direntry: direntry.path,
                         os.scandir(spectrogram_train_clean)))

    X_test_paths = sorted(map(lambda direntry: direntry.path,
                              os.scandir(spectrogram_test_noisy)))
    y_test_paths = sorted(map(lambda direntry: direntry.path,
                              os.scandir(spectrogram_test_clean)))

    train_generator = Generator(X_paths, y_paths, batch_size=batch_size)
    test_generator = Generator(X_test_paths, y_test_paths, batch_size=batch_size) if validate else None

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.Session(config=config)

    callbacks = []
    if log_dir is not None:
        log_dir = "logs/model-{0}".format(int(time.time()))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
            log_dir, "model-{0}".format(time.time())), update_freq='epoch', write_graph=True, profile_batch=0)
        callbacks.append(tensorboard_callback)

    if checkpoint_dir is not None:
        name = "model-cp-epoch_{epoch:04d}.h5"
        path = os.path.join(checkpoint_dir, name)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path, verbose=1, monitor='val_loss', save_best_only=False, mode='auto', period=1)
        callbacks.append(checkpoint)

    
    model = unet(input_size=input_size, loss=loss, optimizer=optimizer["name"], lr=optimizer["lr"])
    model.fit_generator(train_generator, epochs=epochs, shuffle=False, callbacks=callbacks, verbose=1,
                        workers=8, use_multiprocessing=False, validation_data=test_generator, validation_freq=1)


class Generator(tf.keras.utils.Sequence):
    def __init__(self, x_npy_files: List[str], y_npy_files: List[str], batch_size: int):
        self.batch_size = batch_size

        self.sample_count = 0
        for npy_file in x_npy_files:
            shape = np.load(npy_file).shape
            self.sample_count += shape[0]

        self.x_npy_files = x_npy_files
        self.y_npy_files = y_npy_files

        self._on_each_epoch()

    def __len__(self) -> int:
        return int(np.floor(self.sample_count / self.batch_size))

    def __getitem__(self, index):
        if index == 0:
            self._on_each_epoch()
        x_batch = []
        y_batch = []
        for _ in range(0, self.batch_size):
            x_batch.append(next(self.x_generator))
            y_batch.append(next(self.y_generator))
        return np.array(x_batch), np.array(y_batch)

    def _on_each_epoch(self):
        self.x_samples_list = map(np.load, self.x_npy_files)
        self.x_samples_list = map(lambda m: m.reshape(
            m.shape[0], m.shape[1], m.shape[2], 1), self.x_samples_list)

        self.y_samples_list = map(np.load, self.y_npy_files)
        self.y_samples_list = map(lambda m: m.reshape(
            m.shape[0], m.shape[1], m.shape[2], 1), self.y_samples_list)

        def lazy_numpy_vstack(matrices):
            for matrix in matrices:
                yield from matrix

        self.x_generator = lazy_numpy_vstack(self.x_samples_list)
        self.y_generator = lazy_numpy_vstack(self.y_samples_list)
