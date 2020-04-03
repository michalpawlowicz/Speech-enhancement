from generators.SpectogramGenerator import SpectogramGenerator
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from typing import Tuple


class Generator(Sequence):
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
        print("Batch: {0} / {1}".format(index, self.__len__()))
        try:
            sx, sy = self.generator.shape()
            X = np.zeros((self.batch_size, 257, 126))
            Y = np.zeros((self.batch_size, 257, 126))
            print("------------->", X.shape)
            for i in range(self.batch_size):
                x, y = next(self.generator)
                X[i, :, :] = x
                y[i, :, :] = y
            X = X[:, :, :]
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
            Y = Y[:, :, :]
            Y = Y.reshape(Y.shape[0], Y.shape[1], Y.shape[2], 1)
            return X, Y
        except StopIteration:
            raise RuntimeError("Batch index out of range: {0}".format(index))
