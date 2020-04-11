import os

variables = ["TRAIN_CLEAN", "TRAIN_NOISE", "TEST_CLEAN", "TEST_NOISE", "TRAIN_NOISY", "TEST_NOISY", "SAMPLING", "FRAME_LENGTH", "HOP", "STFT_HOP_LENGTH", "N_FFT"]

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