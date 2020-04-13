import os
import numpy as np
import tensorflow as tf
import time
import keras
from environment import check_environment_variables, variables
from model_unet import get_unet, unet

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
checkpoint_name = "cp-{epoch:04d}.h5"
checkpoint_path = os.path.join(env["CHECKPOINTS_DIR"], checkpoint_name)
checkpoint = keras.callbacks.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto', period=1)
log_dir = "logs/model-{}".format(int(time.time()))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_graph=True, profile_batch=0)

model = unet()
model.fit(X, y, epochs=10, batch_size=64,
          shuffle=True, callbacks=None, verbose=1)
