import os

variables = ["INPUT_TRAIN_CLEAN",
             "INPUT_TRAIN_NOISE",
             "INPUT_TEST_CLEAN",
             "INPUT_TEST_NOISE",
             "TRAIN_NOISY",
             "TEST_NOISY",
             "TRAIN_CLEAN",
             "TEST_CLEAN",
             "SAMPLING",
             "FRAME_LENGTH",
             "HOP",
             "STFT_HOP_LENGTH",
             "N_FFT",
             "CHECKPOINTS_DIR",
             "BATCH_SIZE",
             "SAMPLIFY_TRAIN_CLEAN",
             "SAMPLIFY_TRAIN_NOISY",
             "SAMPLIFY_TEST_CLEAN",
             "SAMPLIFY_TEST_NOISY",
             "SPECTROGRAM_TRAIN_CLEAN",
             "SPECTROGRAM_TRAIN_NOISY",
             "SPECTROGRAM_TEST_CLEAN",
             "SPECTROGRAM_TEST_NOISY",
             "SAMPLIFY_NPY_SIZE",
             "PRED_IN_AUDIO",
             "PRED_OUT_AUDIO",
             "MODEL"]


def check_environment_variables(variables):
    env = dict()
    for variable in variables:
        if variable not in os.environ:
            print("Variable {} is missing".format(variable))
        else:
            env[variable] = os.environ[variable]
    if len(variables) != len(env):
        return None

    print("Variables:")
    for variable in env:
        print("    ", variable, env[variable])
    return env
