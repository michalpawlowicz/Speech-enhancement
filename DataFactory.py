import os
import librosa
import itertools
import numpy as np
import random
import sys
from progress.bar import Bar
import pickle
import keras
import matplotlib.pyplot as plt
import librosa.display

from generators.utils import read_sample
from generators.AudioFramesGenerator import AudioFramesGenerator
from generators.AudioGenerator import AudioGenerator
from generators.InputAudioGenerator import InputAudioGererator
from generators.SpectogramGenerator import SpectogramGenerator
from generators.KerasSpectogramGenerator import Generator

def read_files(audio_files, sampling, frame_lenght, hop):
    audio_stacks = []
    bar = Bar('Processing', max=len(audio_files))
    for f in audio_files:
        audio_stacks.append(read_sample(
            sample_path=f, sampling=sampling, frame_lenght=frame_lenght, hop=hop))
        bar.next()
    bar.finish()
    return np.vstack(audio_stacks)


def blend(speech_frames, noise_frames, frame_lenght):

    noise = np.zeros((speech_frames.shape[0], frame_lenght))
    speech = np.zeros((speech_frames.shape[0], frame_lenght))
    noisy_speech = np.zeros((speech_frames.shape[0], frame_lenght))

    bar = Bar('Blending', max=speech_frames.shape[0])
    for i in range(speech_frames.shape[0]):
        noise_magnitute = np.random.uniform(0.2, 0.7)
        noise[i, :] = noise_magnitute * \
            noise_frames[np.random.randint(0, noise_frames.shape[0]), :]
        speech[i, :] = speech_frames[i, :]
        noisy_speech[i, :] = speech[i, :] + noise[i, :]
        bar.next()
    bar.finish()
    return noise, speech, noisy_speech


def get_samples_number(speech_dir, frame_lenght, hop, sampling):
    speech_files = [os.path.join(speech_dir, f)
                    for f in os.scandir(speech_dir)]
    count = 0
    for f in speech_files:
        count += len(read_sample(f, sampling, frame_lenght, hop))
    return count


def create(noise_dir, speech_dir, out_dir, frame_lenght, hop, sampling):
    print("-> Reading noise files")
    noice_files = [os.path.join(noise_dir, f)
                   for f in os.scandir(noise_dir)][:10]

    print("-> Reading speech files")
    speech_files = [os.path.join(speech_dir, f)
                    for f in os.scandir(speech_dir)][:10]

    print("-> Shuffling noise files")
    random.shuffle(noice_files)
    print("-> Shuffling speech files")
    random.shuffle(speech_files)

    print("-> Reading noise into memory")
    noise_frames = read_files(noice_files, sampling, frame_lenght, hop)

    bar = Bar('Blending', max=len(speech_files))
    for sample_file in speech_files:
        y, _ = librosa.load(sample_file, sr=sampling)
        y = np.vstack([y[s:s + frame_lenght]
                       for s in range(0, len(y) - frame_lenght + 1, frame_lenght)])
        indexes = np.random.randint(0, noise_frames.shape[0], y.shape[0])
        magnitutes = np.random.uniform(.2, .6, y.shape[0])
        for i, index in enumerate(indexes):
            y[i, :] = y[i, :] + magnitutes[i] * noise_frames[index, :]
        filename = os.path.splitext(os.path.basename(sample_file))[0] + ".wav"
        librosa.output.write_wav(os.path.join(
            out_dir, filename), y.reshape(1, -1)[0], sr=sampling)
        bar.next()
    bar.finish()


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


def count_samples(audio_files, sampling, frame_lenght, hop):
    count = 0
    for f in audio_files:
        count += read_sample(f, sampling, frame_lenght, hop).shape[0]
    return count


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

