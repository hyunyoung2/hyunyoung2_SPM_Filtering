#!/usr/bin/env bash

# Author: hyunyoung2

#LightNVM 

import numpy as np 

def changeFormatOfDataForSVMLight(a_list, spm) :

    for idx1, var1 in enumerate(a_list) :
        temp_str = str()
        temp = var1.split()
        for idx2, var2 in enumerate(temp) :
            temp_var = float(var2)
            if spm == True : 
                if idx2 == 0 :
                    temp_str += ("-1\t"+str(idx2+1)+":"+str(temp_var)+"\t")
                elif idx2 == (len(a_list)-1) :
                    temp_str += (str(idx2+1)+":"+str(temp_var))
                else :
                    temp_str += (str(idx2+1)+":"+str(temp_var)+"\t")
            else :
                if idx2 == 0 :
                    temp_str += ("+1\t"+str(idx2+1)+":"+str(temp_var)+"\t")
                elif idx2 == (len(a_list)-1) :
                    temp_str += (str(idx2+1)+":"+str(temp_var))
                else :
                    temp_str += (str(idx2+1)+":"+str(temp_var)+"\t")
        a_list[idx1] = temp_str
    return a_list
    

def WriteArrayForSVM_light(file_handler, a_list) :
    for idx, var in enumerate(a_list) :
            if idx <= 10 :
                print (idx)
            file_handler.write(var+"\n")
            
# @ function : Read a file without blank line
# For example : 
# ["Hellow world\n", "\n", "Hi\n"] --> ["Hellow world\n", "Hi\n"]
# input : file path 
# output : a set of Lists of Words
def readAFileLineByLine (file_path) :
    f = open(file_path, "r")
    # due to iconv not dealing with conversion between EUC-KR and UTF-8
    a_whole_list = [x for x in f.readlines() if x != "\n"]
    f.close()
    # For Debugging
    print ("Currently, The file(" + file_path, ") read")    
    return a_whole_list


CBOW_testSen2Vec = ["./data/Vector/test/CBOW_HAMVector",
             "./data/Vector/test/CBOW_SPMVector"]
CBOW_trainSen2Vec = ["./data/Vector/train/CBOW_HAMvector",
              "./data/Vector/train/CBOW_SPMVector"]

SKIP_GRAM_testSen2Vec =["./data/Vector/test/SKIP_GRAM_HAMVector",
             "./data/Vector/test/SKIP_GRAM_SPMVector"]
SKIP_GRAM_trainSen2Vec = ["./data/Vector/train/SKIP_GRAM_HAMvector",
              "./data/Vector/train/SKIP_GRAM_SPMVector"]


# To store it in the 
CBOW_SVM_Lihgt_test = "./data/SVM_Light/test/CBOW_SVM_Light_test"

CBOW_SVM_Lihgt_train = "./data/SVM_Light/train/CBOW_SVM_Light_train"
                              
SKIP_GRAM_SVM_Lihgt_test = "./data/SVM_Light/test/SKIP_GRAM_SVM_Light_test"

SKIP_GRAM_SVM_Lihgt_train = "./data/SVM_Light/train/SKIP_GRAM_SVM_Light_train"
                     

def main(test_ham, test_spm, train_ham, train_spm, target_train, target_test) :
    path1 = test_ham
    path2 = test_spm
    path3 = train_ham
    path4 = train_spm
    
    #for SVM Light 
    svm_light_train=target_train
    svm_light_test=target_test
   
    # for Test
    test_ham_list=readAFileLineByLine(path1)
    test_ham_svm = changeFormatOfDataForSVMLight(test_ham_list, False)
    
    print ("len of test_ham_svm:", len(test_ham_svm))
    
    test_spm_list=readAFileLineByLine(path2)
    test_spm_svm = changeFormatOfDataForSVMLight(test_spm_list, True)
    
    print ("len of test_spm_svm:", len(test_spm_svm))
    
    test_svm = test_ham_svm + test_spm_svm 
    print ("len of test_svm:", len(test_svm))
    print (test_svm[-1])
    with open(svm_light_test, "w") as svm_light_test_writer : 
            WriteArrayForSVM_light(svm_light_test_writer, test_svm)
    
    # for train
    train_ham_list=readAFileLineByLine(path3)
    train_ham_svm = changeFormatOfDataForSVMLight(train_ham_list, False)
    
    print ("len of train_ham_svm:", len(train_ham_svm))
    
    train_spm_list=readAFileLineByLine(path4)
    train_spm_svm = changeFormatOfDataForSVMLight(train_spm_list, True)
    
    print ("len of train_spm_svm:", len(train_spm_svm))  

    train_svm = train_ham_svm + train_spm_svm
    print ("len of train_svm:", len(train_svm)) 
    print (train_svm[-1])
    with open(svm_light_train, "w") as svm_light_train_writer : 
            WriteArrayForSVM_light(svm_light_train_writer, train_svm)
    
    print (len(test_svm))
    print (len(train_svm))
    
    
if __name__ == "__main__" :
    main(CBOW_testSen2Vec[0], CBOW_testSen2Vec[1], CBOW_trainSen2Vec[0], CBOW_trainSen2Vec[1], CBOW_SVM_Lihgt_train, CBOW_SVM_Lihgt_test)
    main(SKIP_GRAM_testSen2Vec[0], SKIP_GRAM_testSen2Vec[1], SKIP_GRAM_trainSen2Vec[0], SKIP_GRAM_trainSen2Vec[1], SKIP_GRAM_SVM_Lihgt_train, SKIP_GRAM_SVM_Lihgt_test)
