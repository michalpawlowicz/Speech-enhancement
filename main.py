import argparse
import os
import sys
from preproces_data import preprocess_data_entry
from train import train_entry
from predict import predict_entry

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
                          default=64, type=int, help='Training batch size')

    data_generation.add_argument('--input-train-clean', dest='input_train_clean',
                                 type=str, help='Directory path to clean speech used as training set')
    data_generation.add_argument('--input-train-noise', dest='input_train_noise',
                                 type=str, help='Directory path to noise used to prepare noisy training set')
    data_generation.add_argument('--input-test-clean', dest='input_test_clean',
                                 type=str, help='Directory path to clean speech used as test set')
    data_generation.add_argument('--input-test-noise', dest='input_test_noise',
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

    if args["mode"] not in ["TRAIN", "GENERATE", "PREDICT"]:
        print("Invalid mode")
        sys.exit(1)

    def check_parameters(vars):
        any_missing = False
        for var in vars:
            if var not in args:
                any_missing = True
                print("Missing parameter: ", var)
        if any_missing:
            sys.exit(1)

    if args["mode"] == "GENERATE":
        check_parameters(["root_dir", "intput_train_clean", "intput_train_noise",
                          "intput_test_clean", "intput_test_noise", "samplify_npy_size"])
        preprocess_data_entry(**args)
    elif args["mode"] == "TRAIN":
        check_parameters(["batch_size", "root_dir"])
        train_entry(**args)
    elif args["mode"] == "PREDICT":
        check_parameters(["model", "in_predict", "out_predict"])
        predict_entry(**args)

