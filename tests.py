import unittest
import numpy as np
from generators.preprocess.Preprocess import insert_frames_into_audio_matrix

####################
M = np.zeros((7, 7))
start_row = 1
F = np.arange(1, 22).reshape(3, -1)
rest = insert_frames_into_audio_matrix(M, F, start_row)

print(M)
print(rest)
####################



M = np.zeros((7, 7))
start_row = 2
F = np.arange(1, 22).reshape(3, -1)
rest = insert_frames_into_audio_matrix(M, F, start_row)
print(M)
print(rest)

M = np.zeros((7, 7))
start_row = 5
F = np.arange(1, 22).reshape(3, -1)
rest = insert_frames_into_audio_matrix(M, F, start_row)
print(M)
print(rest)

M = np.zeros((7, 7))
start_row = 5
F = np.arange(1, 29).reshape(4, -1)
rest = insert_frames_into_audio_matrix(M, F, start_row)
print(M)
print(rest)