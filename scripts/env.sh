#!/bin/bash

if [ -z "$ROOT" ]; then
	echo "Set ROOT variable"
	exit 1
fi

rm -rf $ROOT

mkdir -p $ROOT/Train/sound
mkdir -p $ROOT/Test/sound

mkdir -p $ROOT/Train/samplify/clean
mkdir -p $ROOT/Train/samplify/noisy
mkdir -p $ROOT/Test/samplify/clean
mkdir -p $ROOT/Test/samplify/noisy

mkdir -p $ROOT/Train/spectrogram/clean
mkdir -p $ROOT/Train/spectrogram/noisy
mkdir -p $ROOT/Test/spectrogram/clean
mkdir -p $ROOT/Test/spectrogram/noisy

tree $ROOT
