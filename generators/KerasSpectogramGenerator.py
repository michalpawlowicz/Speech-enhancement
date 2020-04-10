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
        self.input_shape = (generator.shape()[0], generator.shape()[1], 1)

    def __len__(self):
        return int(np.ceil(self.samples_nb / float(self.batch_size)))

    def __getitem__(self, index):
        print("Batch: {0} / {1}".format(index, self.__len__()))
        try:
            X = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1]))
            Y = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1]))
            for i in range(self.batch_size):
                x, y = next(self.generator)
                X[i, :, :] = x
                Y[i, :, :] = y
            X = X[:, :, :]
            X = X.reshape(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            Y = Y[:, :, :]
            Y = Y.reshape(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            return X, Y
        except StopIteration:
            raise RuntimeError("Batch index out of range: {0}".format(index))
