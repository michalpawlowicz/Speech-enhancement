import os
import numpy as np
import tensorflow as tf
import time
import keras
from environment import check_environment_variables, variables
from model import get_unet, unet

env = check_environment_variables(variables)

X = np.load(os.path.join(env["SPECTROGRAM_TRAIN_NOISY"], "0.npy"))
y = np.load(os.path.join(env["SPECTROGRAM_TRAIN_CLEAN"], "0.npy"))

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

print(X.shape)
print(y.shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Save best models to disk during training
base_name = "data_nfft_{0}_fft_hop_{1}_frame_length_{2}".format(env["N_FFT"], env["STFT_HOP_LENGTH"], env["FRAME_LENGTH"])
checkpoint_name = base_name + "-cp-epoch_{epoch:04d}.h5"
checkpoint_path = os.path.join("/home/michal/Documents/Speech-enhancement/checkpoints", checkpoint_name)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=False, mode='auto', period=10)
log_dir = "logs/model-{0}-{1}".format(base_name, int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_graph=True, profile_batch=0)

model = unet(input_size=(128,128,1))
model.fit(X, y, epochs=100, batch_size=128,
          shuffle=False, callbacks=[checkpoint, tensorboard_callback], verbose=1)
