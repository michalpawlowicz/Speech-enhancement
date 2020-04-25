#!/bin/bash

wget http://www.openslr.org/resources/12/dev-clean.tar.gz
wget http://www.openslr.org/resources/12/test-clean.tar.gz
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz

rm -rf LibriSpeech 

function move() {
	ARCH=$1
	DIR=$1
	tar -xf $ARCH
	find $DIR -type f -name '*.flac' | xargs -I {} mv {} $DIR 
	find $DIR -maxdepth 1 -mindepth 1 -type d | xargs rm -r
}

echo "Processing dev-clean"
move dev-clean.tar.gz LibriSpeech/dev-clean/ 

echo "Processing test-clean"
move test-clean.tar.gz LibriSpeech/test-clean/

wget https://github.com/karoldvl/ESC-50/archive/master.zip
7z x master.zip
