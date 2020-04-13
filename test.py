import numpy as np
import librosa
import scipy

frame_size=8192
X = np.random.uniform(1, 100, frame_size)

print(X.shape)
print(librosa.stft(X, n_fft=255, hop_length=64).shape)