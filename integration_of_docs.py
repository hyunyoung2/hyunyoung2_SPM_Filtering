#!/usr/bin/env bash

# Author : hyunyoung2 

# Author: hyunyoung2

#path : 0 -> ham, 1 -> spm
testPath=["./data/Text/test/test_HAM-asp-utf8-5000aa",
        "./data/Text/test/test_SPM-asp-utf8-5000aa"]
trainPath=["./data/Text/train/train_20170616-asp-utf8-5000.ham_aa",
        "./data/Text/train/train_20170616-asp-utf8-5000.spm_aa"]

destination="/home/hyunyoung2/My_lab/smart_conference/data/Text/whole_text"

def readAFileWithoutNewline(file):
    f = open(file, "r")
    
    # Because of iconv program, Error a little happens in converting EUC-KR(or CP949) to UTF-8
    lineByLine = [x for x in f.readlines() if x != "\n"]    
    
    # Remove Unicode space 
    for lineIdx, lineStr in enumerate(lineByLine):
        lineByLine[lineIdx] = " ".join(lineStr.split()) + "\n"

    f.close()
    
    return lineByLine

def checkingBlankLine(file):
    temp=readAFileWithoutNewline(file)
    print(file, ":", len(temp))
    return temp

def main():
    print("checking the number of lines  with newline(EOL)....")
    with open(destination, "w") as wf:
        for p in [testPath, trainPath]:
            for file in p:
                for idx, var in enumerate(checkingBlankLine(file)):
                    wf.write(var)
                
    print("chekcing is all done!!")
    print("checking the integrated docs....")
    print(destination, ":", len(readAFileWithoutNewline(destination)))
        

if __name__ == "__main__":
    main()
