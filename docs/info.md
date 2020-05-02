
## Network's Input dimensions

Required parameters in config.json
	* `input_size (x_nn, y_nn)`: size of spectrogram image, input to NN
	* `fft_hop_length`: fft hop

Parameters infered from required input:
	* ftt's window size will be equal to `x_nn * 2`
	* numper of audio frames is equal to `y_nn`
	* Length of on audio sample is equal to `fft_hop_length * (number of audio frames)` which leads to `fft_hop_length * y_nn`
