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

from generators.preprocess.Utils import read_sample, read_files, count_samples
from generators.AudioFramesGenerator import AudioFramesGenerator
from generators.AudioGenerator import AudioGenerator
from generators.InputAudioGenerator import InputAudioGererator
from generators.SpectogramGenerator import SpectogramGenerator
from generators.KerasSpectogramGenerator import Generator



if "TRAIN_CLEAN" not in os.environ:
    print("Setup TRAIN_CLEAN variable to directory with clean speech")
    sys.exit(1)

if "TRAIN_NOISE" not in os.environ:
    print("Setup TRAIN_NOISE variable to directory with noise")
    sys.exit(2)

if "SAMPLING" not in os.environ:
    print("Setup SAMPLING variable")
    sys.exit(4)

if "FRAME_LENGHT" not in os.environ:
    print("Setup FRAME_LENGHT variable")
    sys.exit(5)

if "HOP" not in os.environ:
    print("Setup HOP variable")
    sys.exit(6)

if "TRAIN_NOISY" not in os.environ:
    print("Setup TRAIN_NOISY variable")
    sys.exit(7)

if "TEST_NOISY" not in os.environ:
    print("Setup TEST_NOISY variable")
    sys.exit(8)

print("Train clean data: ", os.environ["TRAIN_CLEAN"])
print("Train noise data: ", os.environ["TRAIN_NOISE"])
print("Sampling: ", os.environ["SAMPLING"])

sampling = int(os.environ["SAMPLING"])
hop = int(os.environ["HOP"])
frame_lenght = int(os.environ["FRAME_LENGHT"])

# create(noise_dir=os.environ["TRAIN_NOISE"], speech_dir=os.environ["TRAIN_CLEAN"], out_dir=os.environ["TRAIN_NOISY"], frame_lenght=int(
# os.environ["FRAME_LENGHT"]), hop=int(os.environ["HOP"]), sampling=int(os.environ["SAMPLING"]))





# samples_count = count_samples(os.scandir(os.environ["TRAIN_NOISY"]), sampling, frame_lenght, hop)
noisy_files = [d.path for d in os.scandir(os.environ["TRAIN_NOISY"])][:10]
sorted(noisy_files)
clean_files = [d.path for d in os.scandir(os.environ["TRAIN_NOISY"])][:10]
sorted(clean_files)

print(noisy_files)
samples_nb = count_samples(noisy_files, sampling, frame_lenght, hop)
print("Number of samples: ", samples_nb)

if len(noisy_files) != len(clean_files):
    raise RuntimeError("Should be equal")

for x, y in zip(noisy_files, clean_files):
    if x != y:
        raise RuntimeError("Samples are different!")

noisy_audio_generator = AudioGenerator(noisy_files, sampling, frame_lenght, hop)
clean_audio_generator = AudioGenerator(clean_files, sampling, frame_lenght, hop)
input_audio_generatpr = InputAudioGererator(noisy_audio_generator, clean_audio_generator)
spectogram_generator = SpectogramGenerator(input_audio_generatpr, 512)

generator = Generator(64, samples_nb, spectogram_generator)

