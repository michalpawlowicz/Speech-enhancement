import librosa
import numpy as np
from typing import Tuple
from generators.InputAudioGenerator import InputAudioGererator


class SpectogramGenerator():
    def __init__(self, generator: InputAudioGererator, n_fft: int):
        """[summary]

        Arguments:
            generator {InputAudioGererator} -- [description]
            n_fft {int} -- [description]
        """
        self.generator = generator
        self.n_fft = n_fft

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[list, list]:
        """[summary]
        
        Returns:
            Tuple[list, list] -- [description]
        """
        x, y = next(self.generator)
        x_stftaudio = librosa.stft(x, n_fft=self.n_fft)
        x_magnitude, _ = librosa.magphase(x_stftaudio)
        x_magnitude_db = librosa.amplitude_to_db(x_magnitude, ref=np.max)
        y_stftaudio = librosa.stft(y, n_fft=self.n_fft)
        y_magnitude, _ = librosa.magphase(y_stftaudio)
        y_magnitude_db = librosa.amplitude_to_db(y_magnitude, ref=np.max)
        return x_magnitude_db, y_magnitude_db
