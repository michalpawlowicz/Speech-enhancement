from environment import check_environment_variables, variables
import os
from model_unet import get_unet, unet
import numpy as np
import tensorflow as tf

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

model = unet()
model.fit(X, y, epochs=10, batch_size=64, shuffle=True, callbacks=None, verbose=1)