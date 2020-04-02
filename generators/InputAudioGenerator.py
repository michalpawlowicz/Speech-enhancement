from generators.AudioGenerator import AudioGenerator


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

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.noisy_generator)
        y = next(self.clean_generator)
        return x, y
