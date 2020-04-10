import librosa
from collections import Iterator

class AudioFramesGenerator(Iterator):
    """[summary]
    """

    def __init__(self, sample: str, sampling: int, frame_length: int, hop_length: int):
        """[summary]
        
        Arguments:
            sample {str} -- [description]
            sampling {int} -- [description]
            frame_length {int} -- [description]
            hop_length {int} -- [description]
        """

        self.sample = sample
        self.sampling = sampling
        self.frame_length = frame_length
        self.hop_length = hop_length

        y, _ = librosa.load(self.sample, sr=self.sampling)
        self.data = librosa.util.frame(y, frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
        self.iter = iter(self.data)

    def __next__(self):
        return next(self.iter)

    def __iter__(self):
        return self
