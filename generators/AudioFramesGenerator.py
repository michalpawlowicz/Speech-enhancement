from generators.preprocess.Utils import read_sample


class AudioFramesGenerator():
    """[summary]
    """

    def __init__(self, sample: str, sampling: int, frame_lenght: int, hop: int):
        """[summary]
        Arguments:
            sample {str} -- [description]
            sampling {int} -- [description]
            frame_lenght {int} -- [description]
            hop {int} -- [description]
        """
        self.data = read_sample(sample, sampling, frame_lenght, hop)
        self.i = 0

    def __next__(self):
        if self.i < self.data.shape[0]:
            self.i += 1
            return self.data[self.i - 1, :]
        else:
            raise StopIteration()

    def __iter__(self):
        return self
