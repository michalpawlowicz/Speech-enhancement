from environment import check_environment_variables, variables
import librosa
import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler

env = check_environment_variables(variables)

model_path = env["MODEL"]
in_audio_path = env["PRED_IN_AUDIO"]
out_audio_path = env["PRED_OUT_AUDIO"]
sampling = int(env["SAMPLING"])
hop_length = int(env["HOP"])
frame_length = int(env["FRAME_LENGTH"])
n_fft = int(env["N_FFT"])
fft_hop_length = int(env["STFT_HOP_LENGTH"])

y, _ = librosa.load(in_audio_path, sr=sampling)

frames = librosa.util.frame(
    y, frame_length=frame_length, hop_length=hop_length)

spectrograms = []

print(frames.shape)
print(type(frames))

#scaler = MinMaxScaler()
for i in range(frames.shape[0]):
    magnitude, _ = librosa.magphase(librosa.stft(np.asfortranarray(frames[i]), n_fft=n_fft, hop_length=fft_hop_length))
    spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    #scaler.fit(spectrogram)
    #scaler.transform(spectrogram)
    spectrograms.append(spectrogram)

spectrograms = np.array(spectrograms)

print(spectrograms.shape)

model_path = env["MODEL"]

# model.save(model_path)
#model = keras.models.load_model(model_path)
