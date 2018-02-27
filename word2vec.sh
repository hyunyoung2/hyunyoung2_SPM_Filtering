#!/usr/bin/env bash

# Author:hyunyoung2

DIMENSION=300
# 1 CBOW, 0 Skip-gram
CBOW=0
WINDOW=8

ROOT="./data/Vector"

TEXT_ROOT="./data/Text/whole_text"

if [ $CBOW -eq 1 ]; then
	VEC_FILE=${ROOT}"/CBOW_Vector"
	echo "Your vector file is $VEC_FILE"
elif [ $CBOW -eq 0 ]; then
	VEC_FILE=${ROOT}"/SKIP_GRAM_Vector"
        echo "Your vector file is $VEC_FILE"
else 
	echo "you have to configure what type of vector do you want?"
	exit
fi
   
time ./google/trunk/word2vec -train $TEXT_ROOT -output $VEC_FILE  -cbow $CBOW -size $DIMENSION -window $WINDOW -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15

echo "making vector of words is done!!"




