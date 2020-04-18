import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    required = parser.add_argument_group('Required arguments')
    training = parser.add_argument_group('Training')
    data_generation = parser.add_argument_group('Data generation')
    prediction = parser.add_argument_group('Prediction')
    optional = parser.add_argument_group("Optional")

    required.add_argument("--mode", dest="mode", type=str, required=True,
                          help="run in one of three modes [TRAIN | GENERATE | PREDICT]")

    optional.add_argument('--sampling', dest='sampling', default=16000,
                          type=int, help='Audio sampling, default 16000')
    optional.add_argument('--frame-length', dest='frame_length', default=16384,
                          type=int, help='Length of audio passed to NN, default=16384')
    optional.add_argument('--n-fft', dest='n_fft', type=int, default=255,
                          help='n_fft parameter passed to stft, default=255')
    optional.add_argument('--fft-hop', dest='fft_hop', default=128,
                          type=int, help='hop parameter passed to stft, default=128')
    optional.add_argument('--root-dir', dest='root_dir', default="./data", type=str,
                          help='Working directory for storing prepared samples, it should be empty directory, default=./data')

    optional.add_argument('--checkpoints-dir', dest='checkpoints_dir',
                          type=int, help='Directory to store NN checkpoints during training')

    training.add_argument('--batch-size', dest='batch_size',
                          default="batch_size", type=int, help='Training batch size')

    data_generation.add_argument('--input-train-clean', dest='intput_train_clean',
                                 type=str, help='Directory path to clean speech used as training set')
    data_generation.add_argument('--input-train-noise', dest='intput_train_noise',
                                 type=str, help='Directory path to noise used to prepare noisy training set')
    data_generation.add_argument('--input-test-clean', dest='intput_test_clean',
                                 type=str, help='Directory path to clean speech used as test set')
    data_generation.add_argument('--input-test-noise', dest='intput_test',
                                 type=str, help='Directory path to noise used to prepare noisy test set')
    data_generation.add_argument('--samplify-npy-size', dest='samplify_npy_size', type=int,
                                 help='Number of samples in one generated .npy file, smaller for machines with low ram')

    prediction.add_argument('--model', dest='model', type=str,
                            help='Path to model\'s weight when pretrained is used')
    prediction.add_argument('--in-predict', dest='in_predict',
                            type=str, help='Audio path to be denoised')
    prediction.add_argument('--out-predict', dest='out_predict', type=str,
                            help='Path where denoised audio should be saved, path with file name')

    args = parser.parse_args()

    train_noisy = os.path.join(args["root_dir"], "Train", "noisy")
    test_noisy = os.path.join(args["root_dir"], "Test", "noisy")

    train_clean = os.path.join(args["root_dir"], "Train", "clean")
    test_clean = os.path.join(args["root_dir"], "Test", "clean")

    samplify_train_clean = os.path.join(
        args["root_dir"], "Train", "samplify", "clean")
    samplify_test_clean = os.path.join(
        args["root_dir"], "Test", "samplify", "clean")

    samplify_train_noisy = os.path.join(
        args["root_dir"], "Train", "samplify", "noisy")
    samplify_test_noisy = os.path.join(
        args["root_dir"], "Test", "samplify", "noisy")

    spectrogram_train_clean = os.path.join(
        args["root_dir"], "Train", "spectrogram", "clean")
    spectrogram_test_clean = os.path.join(
        args["root_dir"], "Test", "spectrogram", "clean")

    spectrogram_train_noisy = os.path.join(
        args["root_dir"], "Train", "spectrogram", "noisy")
    spectrogram_test_noisy = os.path.join(
        args["root_dir"], "Test", "spectrogram", "noisy")
