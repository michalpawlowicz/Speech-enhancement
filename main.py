import argparse
import sys
import json
import os

from speech_enhancement.preprocess_data import preprocess_data_entry
from speech_enhancement.train import train_entry
from speech_enhancement.predict import predict_entry


def build_env(workdir):
    os.makedirs(os.path.join(workdir, "Train", "noisy"))
    os.makedirs(os.path.join(workdir, "Train", "clean"))
    os.makedirs(os.path.join(workdir, "Test", "noisy"))
    os.makedirs(os.path.join(workdir, "Test", "clean"))

    os.makedirs(os.path.join(workdir, "Train", "samplify", "clean"))
    os.makedirs(os.path.join(workdir, "Train", "samplify", "noisy"))
    os.makedirs(os.path.join(workdir, "Test", "samplify", "clean"))
    os.makedirs(os.path.join(workdir, "Test", "samplify", "noisy"))

    os.makedirs(os.path.join(workdir, "Train", "spectrogram", "clean"))
    os.makedirs(os.path.join(workdir, "Train", "spectrogram", "noisy"))
    os.makedirs(os.path.join(workdir, "Test", "spectrogram", "clean"))
    os.makedirs(os.path.join(workdir, "Test", "spectrogram", "noisy"))

    os.makedirs(os.path.join(workdir, "checkpoints"))
    os.makedirs(os.path.join(workdir, "logs"))


def config_preproces(config):
    def _preprocess(config, workdir):
        for (key, val) in config.items():
            if isinstance(val, str):
                config[key] = os.path.normpath(
                    val.replace("{workdir}", workdir))
            elif isinstance(val, dict):
                config[key] = _preprocess(val, workdir)
        return config
    arr = config['input_size']
    config['input_size'] = (arr[0], arr[1])
    return _preprocess(config, config['workdir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Required arguments')
    required.add_argument("--train", dest="train",
                          required=False, default=False, action="store_true", help="Start training")
    required.add_argument("--gen", dest="gen",
                          required=False, default=False, action='store_true', help="Generate data")
    required.add_argument("--pred", dest="pred",
                          required=False, default=False, action='store_true', help="Denoise input audio")
    required.add_argument("--config", dest="config", type=str,
                          required=True, help="Use given config file")
    required.add_argument("--build-env", dest="build",
                          required=False, default=False, action='store_true', help="Build working dir")
    optional.add_argument("--learning-rate", dest="lr", type=float,
                          required=False, action='store', help="Learning rate")
    args = parser.parse_args()

    if len(list(filter(lambda x: x, [args.train, args.gen, args.pred, args.build]))) != 1:
        parser.print_help()
        sys.exit(1)

    with open(args.config, 'r') as f:
        config = json.load(f)
    config = config_preproces(config)

    if args.train:
        if args.lr is not None:
            print("Overriding learning rate from config file with {}".format(args.lr))
            config["train"]["optimizer"]["lr"] = args.lr
        args = {
            "workdir": config["workdir"],
            "logs": config["train"]["logs"],
            "checkpoints": config["train"]["checkpoints"] if "checkpoints" in config["train"] else None,
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "input_size": (config["input_size"][0], config["input_size"][1], 1),
            "validate": config["train"]["validate"],
            "optimizer": config["train"]["optimizer"],
            "loss": config["train"]["loss"],
            "shuffle": config["train"]["shuffle"] if "shuffle" in config["train"] else False
        }
        train_entry(**args)
    elif args.gen:
        args = {
            "sampling": config["sampling"],
            "frame_length": config["frame_length"],
            "workdir": config["workdir"],
            "n_fft": config["n_fft"],
            "fft_hop_length": config["fft_hop_length"],
            "samplify_npy_size": config["generate"]["samplify_npy_size"],
            "train": config["generate"]["train"],
            "test": config["generate"]["test"]
        }
        preprocess_data_entry(**args)
    elif args.pred:
        args = {
            "sampling": config["sampling"],
            "frame_length": config["frame_length"],
            "n_fft": config["n_fft"],
            "fft_hop_length": config["fft_hop_length"],
            "model": config["predict"]["model"],
            "pred_in_audio": config["predict"]["in_predict"],
            "pred_out_audio": config["predict"]["out_predict"]
        }
        predict_entry(**args)
    elif args.build:
        build_env(config["workdir"])
