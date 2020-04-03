from generators.AudioGenerator import AudioGenerator
from typing import Tuple


class InputAudioGererator():
    """[summary]
    """

    def __init__(self, noisy_generator: AudioGenerator, clean_generator: AudioGenerator):
        """[summary]

        Arguments:
            noisy_generator {AudioGenerator} -- [description]
            clean_generator {AudioGenerator} -- [description]
        """
        self.noisy_generator = noisy_generator
        self.clean_generator = clean_generator

    def shape(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """[summary]
        
        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]] -- [description]
        """
        return self.noisy_generator.shape(), self.clean_generator.shape()

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.noisy_generator)
        y = next(self.clean_generator)
        return x, y
