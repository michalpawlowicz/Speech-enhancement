#!/bin/bash

#wget http://www.openslr.org/resources/12/dev-clean.tar.gz
#wget http://www.openslr.org/resources/12/dev-other.tar.gz
#wget http://www.openslr.org/resources/12/test-clean.tar.gz
#wget http://www.openslr.org/resources/12/test-other.tar.gz
#wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
#wget http://www.openslr.org/resources/12/train-clean-360.tar.gz
#wget http://www.openslr.org/resources/12/train-other-500.tar.gz

rm -rf LibriSpeech 

function convert() {
	DIR=$1
	for i in `find $DIR -type f -name '*.flac'`
	do
		filename="${i%.*}"
		echo $filename
		filename="${filename%%.*}"
		ffmpeg -i $i /tmp/tmp-convert.wav &> /dev/null
		if [ $? -ne 0 ]; then
        		echo "Converting error"
    		fi
       		yes | mv /tmp/tmp-convert.wav $filename.wav
		if [ $? -ne 0 ]; then
        		echo "Moving error"
    		fi
		yes | rm $filename.flac
	done
}

echo "1. dev-clean"
tar -xf dev-clean.tar.gz
find LibriSpeech/dev-clean/ -type f -name '*.flac' | xargs -I {} mv {} LibriSpeech/dev-clean/
find LibriSpeech/dev-clean/ -maxdepth 1 -mindepth 1 -type d | xargs rm -r
convert LibriSpeech/dev-clean

echo "2. dev-other"
tar -xf dev-other.tar.gz
find LibriSpeech/dev-other/ -type f -name '*.flac' | xargs -I {} mv {} LibriSpeech/dev-other/
find LibriSpeech/dev-other/ -maxdepth 1 -mindepth 1 -type d | xargs rm -r
convert LibriSpeech/dev-other

echo "3. test-clean"
tar -xf test-clean.tar.gz
find LibriSpeech/test-clean/ -type f -name '*.flac' | xargs -I {} mv {} LibriSpeech/test-clean/
find LibriSpeech/test-clean/ -maxdepth 1 -mindepth 1 -type d | xargs rm -r
convert LibriSpeech/test-clean

echo "4. test-other"
tar -xf test-other.tar.gz
find LibriSpeech/test-other/ -type f -name '*.flac' | xargs -I {} mv {} LibriSpeech/test-other/
find LibriSpeech/test-other/ -maxdepth 1 -mindepth 1 -type d | xargs rm -r
convert LibriSpeech/test-other


