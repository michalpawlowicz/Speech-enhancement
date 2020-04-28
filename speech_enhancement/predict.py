import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def magnitude_db_and_phase_to_audio(frame_length, fft_hop_length, magnitude_db, phase):
    amplitude = librosa.db_to_amplitude(magnitude_db, ref=1.0)
    return librosa.core.istft(amplitude * phase, hop_length=fft_hop_length, length=frame_length)


def predict_entry(**kwargs):
    model_path = kwargs["model"]
    sampling = kwargs["sampling"]
    frame_length = kwargs["frame_length"]
    n_fft = kwargs["n_fft"]
    fft_hop_length = kwargs["fft_hop_length"]
    in_audio_path_v = kwargs["pred_in_audio"]
    out_audio_path_v = kwargs["pred_out_audio"]
    scaler_path = kwargs["scaler_path"]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config=config)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    for in_audio_path, out_audio_path in zip(in_audio_path_v, out_audio_path_v):
        y, _ = librosa.load(in_audio_path, sr=sampling)
        audio_frames = librosa.util.frame(
            y, frame_length=frame_length, hop_length=frame_length, axis=0)

        X_spectrograms_prenormalized = []
        X_spectrograms = []
        phases = []
        for i in range(audio_frames.shape[0]):
            magnitude, phase = librosa.magphase(librosa.stft(
                np.asfortranarray(audio_frames[i]), n_fft=n_fft, hop_length=fft_hop_length))
            spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
            X_spectrograms_prenormalized.append(spectrogram)
            shape = spectrogram.shape
            spectrogram = scaler.transform(
                spectrogram.reshape(1, -1)).reshape(shape)
            X_spectrograms.append(spectrogram)
            phases.append(phase)

        X_spectrograms_prenormalized = np.array(X_spectrograms_prenormalized)
        X_spectrograms = np.array(X_spectrograms)
        X_spectrograms = X_spectrograms.reshape(
            X_spectrograms.shape[0], X_spectrograms.shape[1], X_spectrograms.shape[2], 1)

        model = tf.keras.models.load_model(model_path)
        Y = model.predict(X_spectrograms)[:, :, :, 0]

        for i, y in enumerate(Y):
            Y[i:] = scaler.inverse_transform(y.reshape(1, -1)).reshape(shape)

        audio = np.zeros((Y.shape[0], frame_length))
        for i, (x, y, p) in enumerate(zip(X_spectrograms_prenormalized, Y, phases)):
            audio[i, :] = magnitude_db_and_phase_to_audio(
                frame_length, fft_hop_length, y - x, p)

        librosa.output.write_wav(out_audio_path, audio.reshape(1, -1)[0], sampling, norm=True)
