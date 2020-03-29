import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from model_unet import unet
from data_tools import scaled_in, scaled_ou
import os
import time
import datetime


def training(path_save_spectrogram, weights_path, name_model, training_from_scratch, epochs, batch_size):
    """ This function will read noisy voice and clean voice spectrograms created by data_creation mode,
    and train a Unet model on this dataset for epochs and batch_size specified. It saves best models to disk regularly
    If training_from_scratch is set to True it will train from scratch, if set to False, it will train
    from weights (name_model) provided in weights_path
    """
    # load noisy voice & clean voice spectrograms created by data_creation mode
    X_in = np.load(path_save_spectrogram + 'noisy_voice_amp_db'+".npy")
    X_ou = np.load(path_save_spectrogram + 'voice_amp_db'+".npy")
    # Model of noise to predict
    X_ou = X_in - X_ou

    # Check distribution
    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    # to scale between -1 and 1
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    # Check shape of spectrograms
    print(X_in.shape)
    print(X_ou.shape)
    # Check new distribution
    print(stats.describe(X_in.reshape(-1, 1)))
    print(stats.describe(X_ou.reshape(-1, 1)))

    # Reshape for training
    X_in = X_in[:, :, :]
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    X_ou = X_ou[:, :, :]
    X_ou = X_ou.reshape(X_ou.shape[0], X_ou.shape[1], X_ou.shape[2], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_in, X_ou, test_size=0.10, random_state=42)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # If training from scratch
    if training_from_scratch:
        generator_nn = unet()
    # If training from pre-trained weights
    else:
        generator_nn = unet(pretrained_weights=weights_path+name_model+'.h5')
    # Save best models to disk during training
    # checkpoint_name = "cp-{epoch:04d}.h5"
    # checkpoint_path = os.path.join(weights_path, checkpoint_name)
    # checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss',save_best_only=True, mode='auto', period=1)

    #log_dir = "logs/model-{}".format(int(time.time()))
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Training
    generator_nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[
                     tensorboard_callback], verbose=1, validation_data=(X_test, y_test))
