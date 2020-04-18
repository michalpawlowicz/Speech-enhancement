from environment import check_environment_variables, variables
import librosa
import tensorflow.keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

env = check_environment_variables(variables)

model_path = env["MODEL"]
in_audio_path = env["PRED_IN_AUDIO"]
out_audio_path = env["PRED_OUT_AUDIO"]
sampling = int(env["SAMPLING"])
frame_length = int(env["FRAME_LENGTH"])
n_fft = int(env["N_FFT"])
fft_hop_length = int(env["STFT_HOP_LENGTH"])

y, _ = librosa.load(in_audio_path, sr=sampling)

frames = librosa.util.frame(
    y, frame_length=frame_length, hop_length=frame_length, axis=0)

X = []
phases = []

print(frames.shape)
print(type(frames))


scalers = []

for i in range(frames.shape[0]):
    scaler = MinMaxScaler()
    scalers.append(scaler)
    magnitude, phase = librosa.magphase(librosa.stft(
        np.asfortranarray(frames[i]), n_fft=n_fft, hop_length=fft_hop_length))
    spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    scaler.fit(spectrogram)
    spectrogram = scaler.transform(spectrogram)
    X.append(spectrogram)
    phases.append(phase)


X = np.array(X)
print(X.shape)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_path = env["MODEL"]
print("Loading model: ", model_path)
model = tensorflow.keras.models.load_model(model_path)

Y = []

Y = model.predict(X)[:, :, :, 0]
scaler = MinMaxScaler()

for i, y in enumerate(Y):
    Y[i, :, :] = scalers[i].inverse_transform(y)


def magnitude_db_and_phase_to_audio(frame_length, hop_length_fft, stftaudio_magnitude_db, stftaudio_phase):
    stftaudio_magnitude_rev = librosa.db_to_amplitude(
        stftaudio_magnitude_db, ref=1.0)
    audio_reverse_stft = stftaudio_magnitude_rev * stftaudio_phase
    audio_reconstruct = librosa.core.istft(
        audio_reverse_stft, hop_length=hop_length_fft, length=frame_length)
    return audio_reconstruct


audio = np.zeros((Y.shape[0], frame_length))

for i, (y, p) in enumerate(zip(Y, phases)):
    audio[i, :] = magnitude_db_and_phase_to_audio(
        frame_length, fft_hop_length, y, p)

audio = audio.reshape(1, -1)
print(audio.shape)
print(audio)

librosa.output.write_wav(out_audio_path, audio[0], sampling, norm=True)
