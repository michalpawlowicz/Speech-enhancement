from model_unet import unet, get_unet
import datetime
import os
import librosa
import itertools
import numpy as np
import random
import sys
import pickle
import keras
import matplotlib.pyplot as plt
import librosa.display
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

from generators.preprocess.Utils import read_sample, read_files, count_samples
from generators.AudioFramesGenerator import AudioFramesGenerator
from generators.AudioGenerator import AudioGenerator
from generators.InputAudioGenerator import InputAudioGererator
from generators.SpectogramGenerator import SpectogramGenerator
from generators.KerasSpectogramGenerator import Generator
from generators.preprocess.Preprocess import create

from environment import check_environment_variables, variables

env = check_environment_variables(variables)
if env is None:
    raise RuntimeError("Setup variables")

sampling = int(env["SAMPLING"])
frame_length = int(env["FRAME_LENGTH"])
hop = int(env["HOP"])

#create(noise_dir=env["TRAIN_NOISE"], speech_dir=env["TRAIN_CLEAN"], out_dir=env["TRAIN_NOISY"], frame_lenght=frame_length, hop=hop, sampling=sampling)

noisy_files = [d.path for d in os.scandir(env["TRAIN_NOISY"])]
sorted(noisy_files)
noisy_files = noisy_files[:10]
clean_files = [d.path for d in os.scandir(env["TRAIN_CLEAN"])]
sorted(clean_files)
clean_files = clean_files[:10]

if len(noisy_files) != len(clean_files):
    raise RuntimeError("Number of clean samples and noisy samples is different {0} vs. {1}".format(len(noisy_files), len(clean_files)))

for x, y in zip(noisy_files, clean_files):
    if os.path.basename(x) != os.path.basename(y):
        raise RuntimeError("Samples are different! {0} vs {1}".format(x, y))


samples_nb = count_samples(noisy_files, sampling, frame_length, hop)
print("Number of samples: ", samples_nb)

n_fft = int(env["N_FFT"])
fft_hop = int(env["STFT_HOP_LENGTH"])

noisy_audio_generator = AudioGenerator(
    noisy_files, sampling, frame_length, hop)
clean_audio_generator = AudioGenerator(
    clean_files, sampling, frame_length, hop)
input_audio_generator = InputAudioGererator(
    noisy_audio_generator, clean_audio_generator)
spectogram_generator = SpectogramGenerator(generator=input_audio_generator, n_fft=n_fft, hop_length=fft_hop)

generator = Generator(int(env["BATCH_SIZE"]), samples_nb, spectogram_generator)

model = unet(input_size=generator.input_shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

checkpoint_name = "cp-{epoch:04d}.h5"
checkpoint_path = os.path.join(env["CHECKPOINTS_DIR"], checkpoint_name)
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, monitor='val_loss', save_best_only=True, mode='auto', period=1)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', write_graph=True, profile_batch=0)

model.fit_generator(generator, steps_per_epoch=None, epochs=100, verbose=1, callbacks=[checkpoint, tensorboard_callback], validation_data=None,
                    validation_steps=None, validation_freq=1, workers=8, use_multiprocessing=False, shuffle=False, initial_epoch=0)
