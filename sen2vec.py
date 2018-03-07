#!/usr/bin/env bash

# Author : hyunyoung2

# Author: hyunyoung2

# This code make sentence vector from word vector 

import numpy as np

# for text
# path : 0 -> ham, 1 -> spm
testPath = ["./data/Text/test/test_HAM-asp-utf8-5000aa",
         "./data/Text/test/test_SPM-asp-utf8-5000aa"]
trainPath = ["./data/Text/train/train_20170616-asp-utf8-5000.ham_aa",
         "./data/Text/train/train_20170616-asp-utf8-5000.spm_aa"]

# for vector 
CBOW_testSen2Vec = ["./data/Vector/test/CBOW_HAMVector",
             "./data/Vector/test/CBOW_SPMVector"]
CBOW_trainSen2Vec = ["./data/Vector/train/CBOW_HAMvector",
              "./data/Vector/train/CBOW_SPMVector"]

SKIP_GRAM_testSen2Vec =["./data/Vector/test/SKIP_GRAM_HAMVector",
             "./data/Vector/test/SKIP_GRAM_SPMVector"]
SKIP_GRAM_trainSen2Vec = ["./data/Vector/train/SKIP_GRAM_HAMvector",
              "./data/Vector/train/SKIP_GRAM_SPMVector"]

# for vector of whole text
wholeVector=["./data/Vector/utf8_CBOW_Vector",
            "./data/Vector/utf8_SKIP_GRAM_Vector"]

# read vector file created from whole text 
# The first line is the number of words and dimensionality 
# word vector : hi 1 2 3 1 --> [(hi, 1, 2, 3 , 1), ]
def readWholeVectorFile(file):
    
    wordVec = list ()
    
    with open(file, "r") as rf:
        for x in rf.readlines() :
            if x == "\n" : 
                print ("ERROR-Python : you have the blank line :", file)
                exit()
            wordVec.append(tuple(x.split())) 
            
    # To verify
    print(file, ":", len(wordVec))
    return wordVec



# separate line into word by white space 
# if you have a doc like "i am boy\n, you're girl\n" 
# ----->  [(I, am, boy),(you're,  girl )]
def separateLine(file):
    return readWholeVectorFile(file)



# separate line into word by white space 
# if you have a doc like "i am boy\n, you're girl\n" 
# ----->  [(I, am, boy),(you're,  girl )]
def readAFileWithoutNewline(file):
    
    wordList = list ()
    
    with open(file, "r") as f:
        # Because of iconv program, Error a little happens in converting EUC-KR(or CP949) to UTF-8
        lineByLine = [x for x in f.readlines() if x != "\n"]    

        # Remove Unicode space 
        for lineIdx, lineStr in enumerate(lineByLine):
            wordList.append(tuple(lineStr.split()))
    print(wordList[0:2])
    print(file, ":", len(wordList))
    return wordList

    
# Make hash function of words like {hi:[1,2,3,4]. }
def hashFunctionOfWords(wordList):
    hashWord2Vec = dict()
    
    print("the number of words in a list:", len(wordList))
    
    for idx, var in enumerate(wordList):
        if hashWord2Vec.get(var[0]) == None:
            hashWord2Vec[var[0]] = np.array(list(var[1:]), np.float)
        else:
            print("ERROR-Python, duplication of a word happened")
            exit()
    
    return hashWord2Vec


# Make sentence vector from word2vec using adding. 
def sen2vec(hashWordVec, lineList, dim):
    senVec = list()
    errWord = list()
    
    for lineIdx, lineVar in enumerate(lineList):
        senVec.append(np.zeros((dim,), dtype=np.float))
        for wordIdx, word in enumerate(lineVar):
            try: 
                senVec[lineIdx] += hashWordVec[word]
            except: 
                errWord.append((lineIdx, word))
                
    print("The number of error word:", len(errWord))

    return senVec
# Make TSV flie to utitlize

def tsvFile(file, senVec):
    print("writing", file, "to tsv file .....")
    with open(file, "wb") as wf:
        np.savetxt(wf, senVec,  delimiter="\t")
    print("writing is done!\n\n\n")
    
def main():
    
    # first CBOW, Second Skip_gram
    # wholeVector = [utf8_CBOW_Vector, utf8_SKIP_GRAM_vector]
    for typeIdx, typeName in enumerate(wholeVector):
        print("reading", typeName, ".....")
        totalWordVec = readWholeVectorFile(typeName)
        dim = totalWordVec[0][1]
        print("Dimensionality of a word vector:", dim, "\n")
        print("making hash function of wordvec....")
        hashWord = hashFunctionOfWords(totalWordVec[1:])
       
    
        # train file 
        for trainIdx, trainFile in enumerate(trainPath):
            print("reading", trainFile, "..... and making sen2vec")
            senVec = sen2vec(hashWord, readAFileWithoutNewline(trainFile), int(dim))
            if typeIdx == 0: # CBOW
                continue
                if trainIdx == 0: # ham
                    tsvFile(CBOW_trainSen2Vec[0], senVec)
                else: #spm
                    tsvFile(CBOW_trainSen2Vec[1], senVec)
            else: # Skip gram
                if trainIdx == 0: # ham
                    tsvFile(SKIP_GRAM_trainSen2Vec[0], senVec)
                else: #spm
                    tsvFile(SKIP_GRAM_trainSen2Vec[1], senVec)
                
        # test file  
        for testIdx, testFile in enumerate(testPath):
            print("reading", testFile, "..... and making sen2vec")
            senVec = sen2vec(hashWord, readAFileWithoutNewline(testFile), int(dim))
            if typeIdx == 0: # CBOW
                continue
                if testIdx == 0: # ham
                    tsvFile(CBOW_testSen2Vec[0], senVec)
                else: #spm        
                    tsvFile(CBOW_testSen2Vec[1], senVec)
            else: # Skip gram
                if testIdx == 0: # ham
                    tsvFile(SKIP_GRAM_testSen2Vec[0], senVec)
                else: #spm
                    tsvFile(SKIP_GRAM_testSen2Vec[1], senVec)
    
if __name__ == "__main__":
    main()
