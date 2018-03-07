#!/usr/bin/env bash

# Author : hyunyoung2

ROOT="./tensorflow_log"
CBOW=${ROOT}"/CBOW"
SKIPGRAM=${ROOT}"/SKIPGRAM"


echo "With CBOW..."
echo ""

echo "straring classification of SPM"
echo "layer 1..."
#python3 ./feedforward_NN_layer1_CBOW.py > ${CBOW}"/layer1_log"

echo "layer 2..."
#python3 ./feedforward_NN_layer2_CBOW.py > ${CBOW}"/layer2_log"

echo "layer 3..."
python3 ./feedforward_NN_layer3_CBOW.py > ${CBOW}"/layer3_log"

echo "layer 4..."
python3 ./feedforward_NN_layer4_CBOW.py > ${CBOW}"/layer4_log"

echo "layer 5..."
#python3 ./feedforward_NN_layer5_CBOW.py > ${CBOW}"/layer5_log"


echo "With skip gram..."
echo ""

echo "starting classification of spm"
echo "layer 1..."
#python3 ./feedforward_NN_layer1_Skip_gram.py > ${SKIPGRAM}"/layer1_log"

echo "layer 2..."
#python3 ./feedforward_NN_layer2_Skip_gram.py > ${SKIPGRAM}"/layer2_log"

echo "layer 3..."
python3 ./feedforward_NN_layer3_Skip_gram.py > ${SKIPGRAM}"/layer3_log"

echo "layer 4..."
python3 ./feedforward_NN_layer4_Skip_gram.py > ${SKIPGRAM}"/layer4_log"

echo "layer 5..."
#python3 ./feedforward_NN_layer5_Skip_gram.py > ${SKIPGRAM}"/layer5_log"


echo "neural network is done!!"


