#!/usr/bin/env bash

# Author : hyunyoung2

import tensorflow as tf
import numpy as np
from tqdm import tqdm 

# CBOW_HAMvector CBOW_SPMVector  SKIP_GRAM_HAMvector  SKIP_GRAM_SPMVector
CBOWTrain = ["./data/Vector/train/CBOW_HAMvector",
            "./data/Vector/train/CBOW_SPMVector",]

SKIPGRAMTrain = ["./data/Vector/train/SKIP_GRAM_HAMvector",
                "./data/Vector/train/SKIP_GRAM_SPMVector"]

# CBOW_HAMVector  CBOW_SPMVector  SKIP_GRAM_HAMVector  SKIP_GRAM_SPMVector
CBOWTest = ["./data/Vector/test/CBOW_HAMVector",
          "./data/Vector/test/CBOW_SPMVector",]


SKIPGRAMTest = ["./data/Vector/test/SKIP_GRAM_HAMVector",
               "./data/Vector/test/SKIP_GRAM_SPMVector"]


# 0 CBOW, 1 SKIPGRAM
VectorType = 0


def tsvToNumpyArr(file, delimiterOfDoc):
    print ("Currently, The file(" + file, ") read")
    return np.genfromtxt(file, delimiter=delimiterOfDoc, dtype=float)

def importCBOWData():
    CBOWTrainHAM = tsvToNumpyArr(CBOWTrain[0], delimiterOfDoc="\t")
    CBOWtrainSPM = tsvToNumpyArr(CBOWTrain[1], delimiterOfDoc="\t")
    CBOWtestHAM = tsvToNumpyArr(CBOWTest[0], delimiterOfDoc="\t")
    CBOWtestSPM = tsvToNumpyArr(CBOWTest[1], delimiterOfDoc="\t")
    return CBOWTrainHAM, CBOWtrainSPM, CBOWtestHAM, CBOWtestSPM
    
def importSKIPGRAMData():
    SKIPGRAMTrainHAM = tsvToNumpyArr(SKIPGRAMTrain[0], delimiterOfDoc="\t")
    SKIPGRAMtrainSPM = tsvToNumpyArr(SKIPGRAMTrain[1], delimiterOfDoc="\t")
    SKIPGRAMtestHAM = tsvToNumpyArr(SKIPGRAMTest[0], delimiterOfDoc="\t")
    SKIPGRAMtestSPM = tsvToNumpyArr(SKIPGRAMTest[1], delimiterOfDoc="\t")
    return SKIPGRAMTrainHAM, SKIPGRAMtrainSPM, SKIPGRAMtestHAM, SKIPGRAMtestSPM


def importData():
    if VectorType == 0:
        CBOWTrainHAM, CBOWtrainSPM, CBOWtestHAM, CBOWtestSPM = importCBOWData()
        return CBOWTrainHAM, CBOWtrainSPM, CBOWtestHAM, CBOWtestSPM
    else:
        SKIPGRAMTrainHAM, SKIPGRAMtrainSPM, SKIPGRAMtestHAM, SKIPGRAMtestSPM = importSKIPGRAMData()
        return SKIPGRAMTrainHAM, SKIPGRAMtrainSPM, SKIPGRAMtestHAM, SKIPGRAMtestSPM
    
trainHAMX, trainSPMX, testHAMX, testSPMX = importData()

print("Importing data.... it is done!")
print("len of trainHAMX", len(trainHAMX), "type of trainHAMX", type(trainHAMX), "shape of trainHAMX", trainHAMX.shape)
print("len of trainSPMX", len(trainSPMX), "type of trainSPMX", type(trainSPMX), "shape of trainSPMX", trainSPMX.shape)
print("len of testHAMX", len(testHAMX), "type of testHAMX", type(testHAMX), "shape of testHAMX", testHAMX.shape)
print("len of testSPMX", len(testSPMX), "type of testSPMX", type(testSPMX), "shape of testSPMX", testSPMX.shape)
print()

def zerosArr(arr):
    return np.zeros((arr.shape[0],1), dtype=float)

def onesArr(arr):
    return np.ones((arr.shape[0],1), dtype=float)

def generateLabel(trainHAMX, trainSPMX, testHAMX, testSPMX):
    trainHAMY = np.concatenate((onesArr(trainHAMX), zerosArr(trainHAMX)), axis=1)
    trainSPMY = np.concatenate((zerosArr(trainSPMX), onesArr(trainSPMX)), axis=1)
    testHAMY = np.concatenate((onesArr(testHAMX), zerosArr(testHAMX)), axis=1)
    testSPMY = np.concatenate((zerosArr(testSPMX), onesArr(testSPMX)), axis=1)
    
    return trainHAMY, trainSPMY, testHAMY, testSPMY
    
trainHAMY, trainSPMY, testHAMY, testSPMY = generateLabel(trainHAMX, trainSPMX, testHAMX, testSPMX)

print("len of trainHAMY", len(trainHAMY), "type of trainHAMY", type(trainHAMY), "shape of trainHAMY", trainHAMY.shape)
print("len of trainSPMY", len(trainSPMY), "type of trainSPMY", type(trainSPMY), "shape of trainSPMY", trainSPMY.shape)
print("len of testHAMY", len(testHAMY), "type of testHAMY", type(testHAMY), "shape of testHAMY", testHAMY.shape)
print("len of testSPMY", len(testSPMY), "type of testSPMY", type(testSPMY), "shape of testSPMY", testSPMY.shape)
print()

def concatenateData(trainHAMX, trainSPMX, testHAMX, testSPMX):
    trainX = np.concatenate((trainHAMX, trainSPMX), axis=0)
    testX = np.concatenate((testHAMX, testSPMX), axis=0)
    
    return trainX, testX


def concatenateLabel(trainHAMY, trainSPMY, testHAMY, testSPMY):
    trainY = np.concatenate((trainHAMY, trainSPMY), axis=0)
    testY = np.concatenate((testHAMY, testSPMY), axis=0)
    
    return trainY, testY



trainX, testX = concatenateData(trainHAMX, trainSPMX, testHAMX, testSPMX)
trainY, testY = concatenateLabel(trainHAMY, trainSPMY, testHAMY, testSPMY)

print(trainX)
print(trainY)

print("Finally, basic setting of data is done!!")

print("len of trainX", len(trainX), "type of trainX", type(trainX), "shape of trainX", trainX.shape)
print("len of testX", len(testX), "type of testX", type(testX), "shape of testX", testX.shape)
print("len of trainY", len(trainY), "type of trainY", type(trainY), "shape of trainY", trainY.shape)
print("len of testY", len(testY), "type of testY", type(testY), "shape of trainY", testY.shape)


# Hyper parameter
EPOCH = 35000
LEARNINGRATE = 0.0008


HIDDENLAYERSIZE = trainX.shape[1] # dimension 
DIMENSION = trainX.shape[1] # demension
LABELSIZE = trainY.shape[1] # true or flase

logs_path = "./log/log_Layer3"

# for input of tensorflow 
input_vectors = tf.placeholder(tf.float64, [None, DIMENSION])
ground_truths = tf.placeholder(tf.float64, [None, LABELSIZE])

weight = { "h1" : tf.get_variable("Layer3_weight_h1", [HIDDENLAYERSIZE, HIDDENLAYERSIZE], dtype=tf.float64),
          "h2" : tf.get_variable("Layer3_weight_h2", [HIDDENLAYERSIZE, HIDDENLAYERSIZE], dtype=tf.float64),
         "out": tf.get_variable("Layer3_weight_output", [HIDDENLAYERSIZE, 2], dtype=tf.float64)}

bias = { "h1" : tf.get_variable("Layer3_bias_h1", [1, HIDDENLAYERSIZE], dtype=tf.float64),
        "h2" : tf.get_variable("Layer3_bias_h2", [1, HIDDENLAYERSIZE], dtype=tf.float64),
         "out" : tf.get_variable("Layer3_bias_output", [1, 2], dtype=tf.float64)}

def FeedforwardNN(input_vector, weight, biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_vectors, weight["h1"]), biases["h1"]))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weight["h2"]), biases["h2"]))
    output = tf.add(tf.matmul(layer2, weight["out"]), bias["out"])
    
    return output

prediction = FeedforwardNN(input_vectors, weight, bias)
    
# If sigmoid_cross_entropy_with_logits is turned into softmax_cross_entropy_with_logits    
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truths, logits=prediction))
tf.summary.scalar("Loss", cost)

trainingOptimzer = tf.train.GradientDescentOptimizer(learning_rate=LEARNINGRATE).minimize(cost)

correctPredictionOP = tf.equal(tf.argmax(prediction,1), tf.argmax(ground_truths,1))
accuracyOP = tf.reduce_mean(tf.cast(correctPredictionOP, tf.float32))
tf.summary.scalar("Accuracy", accuracyOP)

initOP = tf.global_variables_initializer()

# Merge all summaries into a single op
mergedOP = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(initOP)
    
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    
    # training cycle
    for i in tqdm(range(EPOCH)) :
        opt, summary =  sess.run([trainingOptimzer, mergedOP], feed_dict={input_vectors : trainX, ground_truths: trainY})
        # Write logs at every iteration
        if i % 1000 == 0 : 
            summary_writer.add_summary(summary, i)
            train_accuracy, train_cost = sess.run(
                [accuracyOP, cost], 
                feed_dict={input_vectors : trainX, ground_truths: trainY})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, train_cost %g"%(i, train_cost))
    
    # Finally, check Test data
    print ("Final Accuracy on Test set: %s" % str(sess.run(accuracyOP, feed_dict={input_vectors: testX, ground_truths: testY})))

