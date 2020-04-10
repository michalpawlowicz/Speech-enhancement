from generators.AudioFramesGenerator import AudioFramesGenerator
from collections import Iterator


class AudioGenerator(Iterator):
    def __init__(self, samples: list, sampling: int, frame_lenght: int, hop: int):
        """[summary]

        Arguments:
            samples {list} -- [description]
            sampling {int} -- [description]
            frame_lenght {int} -- [description]
            hop {int} -- [description]

        Raises:
            RuntimeError: [description]
        """
        if len(samples) == 0:
            raise RuntimeError("Samples shouldn't be a empty array")
        self.samples = samples
        self.sampling = sampling
        self.frame_lenght = frame_lenght
        self.hop = hop
        self.m = len(self.samples)
        self.i = 0
        self.curr = AudioFramesGenerator(
            self.samples[0], sampling, frame_lenght, hop)

    def __next__(self):
        try:
            return next(self.curr)
        except:
            if self.i < self.m:
                self.curr = AudioFramesGenerator(
                    self.samples[self.i], self.sampling, self.frame_lenght, self.hop)
                self.i += 1
                return self.__next__()
            else:
                raise StopIteration()

    def __iter__(self):
        return self
