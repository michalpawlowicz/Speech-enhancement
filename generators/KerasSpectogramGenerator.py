from SpectogramGenerator import SpectogramGenerator
import keras
import numpy as np


class Generator(keras.utils.Sequence):
    """[summary]
    """

    def __init__(self, batch_size: int, samples_nb: int, generator: SpectogramGenerator):
        """[summary]
        
        Arguments:
            keras {[type]} -- [description]
            batch_size {int} -- [description]
            samples_nb {int} -- [description]
            generator {SpectogramGenerator} -- [description]
        """
        self.batch_size = batch_size
        self.samples_nb = samples_nb
        self.batch_nb = 0
        self.generator = generator

    def __len__(self):
        return int(np.ceil(self.samples_nb / float(self.batch_size)))

    def __getitem__(self, index):
        try:
            x, y = itertools.islice(array, self.batch_size)
            return x, y
        except:
            return [], []
