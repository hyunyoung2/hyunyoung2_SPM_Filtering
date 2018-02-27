#!/usr/bin/env bash

# Author : huynyoung2

echo "starting skip-gram based doc2vec to learn"

#./svm_learn ../../data/SVM_Light/train/SKIP_GRAM_SVM_Light_train  ./SKIP_GRAM_SVM_Light_train_model

echo "starting cbow based doc2vec to learn"

#./svm_learn ../../data/SVM_Light/train/CBOW_SVM_Light_train ./CBOW_SVM_Light_train_model

echo "staring classification sentences based on skip gram"
echo "check the result on SKIP_GRAM_SVM_Light_test_resulr in log dir "
./svm_classify  ../../data/SVM_Light/test/SKIP_GRAM_SVM_Light_test ./SKIP_GRAM_SVM_Light_train_model ./SKIP_GRAM_SVM_Lihgt_Prediction > SKIP_GRAM_SVM_Light_test_result

echo "starting classifciation senetences based on CBOW"
echo "check the result on CBOW_SVM_Light_test_result"

#./svm_classify  ../../data/SVM_Light/test/CBOW_SVM_Light_test  ./CBOW_SVM_Light_train_model ./CBOW_SVM_Light_Predictioni > CBOW_SVM_Light_test_result
