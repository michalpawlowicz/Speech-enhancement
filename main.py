import argparse
import sys
import json

from preprocess_data import preprocess_data_entry
from train import train_entry
from predict import predict_entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--train", dest="train", type=bool,
                          required=False, default=False, help="Start training")
    required.add_argument("--gen", dest="gen", type=bool,
                          required=False, default=False, help="Generate data")
    required.add_argument("--pred", dest="pred", type=bool,
                          required=False, default=False, help="Denoise input audio")
    required.add_argument("--config", dest="config", type=str,
                          required=True, help="Use given config file")
    args = parser.parse_args()

    if len(filter(lambda x: x, [args["train"], args["gen"], args["pred"]])) != 1:
        parser.print_help()
        sys.exit(1)

    with open(args["config"], 'r') as f:
        config = json.load(f)

    if args["train"]:
        args = {"workdir": config["workdir"],
                "logs": config["logs"],
                "checkpoints": config["checkpoints"],
                "epochs": config["epochs"],
                "input_size": config["input_size"]
                }
        train_entry(**args)
    elif args["gen"]:
        args = {
            "sampling": config["sampling"],
            "frame_length": config["frame_length"],
            "workdir": config["workdir"],
            "n_fft": config["n_fft"],
            "fft_hop_length": config["fft_hop_length"],
            "samplify_npy_size": config["samplify_npy_size"],
            "train": config["train"],
            "test": config["test"]
        }
        preprocess_data_entry(**args)
    elif args["pred"]:
        predict_entry(**args)
