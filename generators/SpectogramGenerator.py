import librosa
import numpy as np
from typing import Tuple
from generators.InputAudioGenerator import InputAudioGererator
from collections import Iterator


class SpectogramGenerator(Iterator):
    def __init__(self, generator: InputAudioGererator, n_fft : int = 512, hop_length : int = 64):
        """[summary]

        Arguments:
            generator {InputAudioGererator} -- [description]
            n_fft {int} -- [description]
        """
        self.generator = generator
        self.n_fft = n_fft
        self.hop_length = hop_length

    def shape(self):
        return 257, 251

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[list, list]:
        """[summary]
        
        Returns:
            Tuple[list, list] -- [description]
        """

        x, y = next(self.generator)
        x_stftaudio = librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        x_magnitude, _ = librosa.magphase(x_stftaudio)
        x_magnitude_db = librosa.amplitude_to_db(x_magnitude, ref=np.max)
        y_stftaudio = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        y_magnitude, _ = librosa.magphase(y_stftaudio)
        y_magnitude_db = librosa.amplitude_to_db(y_magnitude, ref=np.max)

        return x_magnitude_db, y_magnitude_db
