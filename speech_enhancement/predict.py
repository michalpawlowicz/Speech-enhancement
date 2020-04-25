import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config=config)

    for in_audio_path, out_audio_path in zip(in_audio_path_v, out_audio_path_v):
        audio_y, _ = librosa.load(in_audio_path, sr=sampling)
        frames = librosa.util.frame(
            audio_y, frame_length=frame_length, hop_length=frame_length, axis=0)

        audio_y = frames.reshape(1, -1)

        X = []
        phases = []
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
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        model = tf.keras.models.load_model(model_path)

        Y = model.predict(X)[:, :, :, 0]

        for i, y in enumerate(Y):
            Y[i, :, :] = scalers[i].inverse_transform(y)

        audio = np.zeros((Y.shape[0], frame_length))

        for i, (y, p) in enumerate(zip(Y, phases)):
            audio[i, :] = magnitude_db_and_phase_to_audio(
                frame_length, fft_hop_length, y, p)

        audio = audio_y - audio.reshape(1, -1)
        #audio = audio.reshape(1, -1)
        librosa.output.write_wav(out_audio_path, audio[0], sampling, norm=True)
