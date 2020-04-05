import librosa

class AudioFramesGenerator():
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
        y, _ = librosa.load(sample, sr=sampling)
        self.data = librosa.utils.frames(y, frame_length=frame_length, hop_length=hop_length, axis=0)
        self.iter = iter(self.data)

    def __next__(self):
        return next(self.iter)

    def __iter__(self):
        return self
