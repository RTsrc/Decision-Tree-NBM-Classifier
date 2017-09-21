import numpy as np
from DTL import getDocMatrix, getWord, getLabelName, getDoc, wordInDoc, getWordCount, getDocNum
import math
import time
from enum import IntEnum

class Labels (IntEnum):
    atheism = 1
    graphics = 2
    
#load all the data into numpy arrays    
trainData = np.loadtxt("trainData.txt")
testData = np.loadtxt("testData.txt")
trainLabel = np.loadtxt("trainLabel.txt")
testLabel = np.loadtxt("testLabel.txt")
file = open("words.txt")
words = file.read().splitlines()
labels = ["alt.atheism", "comp.graphics"]

def getLabelFreq(lbldata):
    docs = np.where(lbldata == Labels.atheism)
    docs2 = np.where(lbldata == Labels.graphics)
    ptotal = docs[0].size
    ntotal = docs2[0].size
    resu = np.array([ptotal, ntotal])
    return resu

#returns the number of times a word appears in a doc
def countWordinDoc(docId, wordId, data):
    return data[docId - 1, wordId -1]

#count all words occurrences in a given dataset
def countWordAll(wordId, data):
    subset = data[:, wordId - 1]
    count = np.sum(subset)
    return count

def getFreqTable(docdata, lbldata):
    wtotal = getWordCount(words)
    f = np.zeros((2, wtotal))
    docs = np.where(lbldata == Labels.atheism)
    docs2 = np.where(lbldata == Labels.graphics)
    docArr = docdata[docs[0], :]
    docArr2 = docdata[docs2[0], :]
    
    for word in range(1, wtotal + 1):
        pcount= countWordAll(word, docArr)
        ncount = countWordAll(word, docArr2)
        f[0, word - 1] = pcount
        f[1, word - 1] = ncount
        
    return f

def getRelFreq(FTable, lbldata):
    rf = np.copy(FTable)
    #normalize and smooth
    p_class = getLabelFreq(lbldata)
    rf[0,:] = (rf[0,:] + 1)/(p_class[0] + p_class.size)
    rf[1,:] = (rf[1,:] + 1)/(p_class[1] + p_class.size)
    
    return rf

def getLogProb(FTable):
    lf = np.copy(FTable)
    
    for data in lf:
        data[...] = np.log(data)
        
    return lf

def getDiscriminality(freqData, wordId):
    p1 = freqData[Labels.atheism - 1, wordId -1]
    p2 = freqData[Labels.graphics - 1, wordId -1]
    #print(p1, p2)
    resu = abs(np.log(p1) - np.log(p2)) 
    return resu

def DiscrimLst(freqData, wordLst):
    dlst = []
    for wordId in wordLst:
        dval = getDiscriminality(freqData, wordId)
        dlst.append(dval)
    return np.asarray(dlst)

def getTopD(dLst, wordLst,n):
    srtLst = np.argsort(dLst)[::-1]
    topWords = wordLst[srtLst]
    PrintWords(topWords[0:n], dLst)
    return topWords[0:n]

def PrintWords(wordLst, dLst):
    print("DVals:", dLst[wordLst - 1])
    for wordId in wordLst:
        print(getWord(wordId),dLst[wordId -1])
        
        
def getDocWords(data, docId):
    resu = np.where(data[docId - 1, :] > 0)
    return resu[0]

def getNDocWords(data, docId):
    resu = np.where(data[docId - 1, :] == 0)
    return resu[0]

def normalize(f):
    return f/np.sum(f)

def calcProb(data, freqT, docId, lbldata, useLog):
    p_class = getLabelFreq(lbldata)
    #normalize and smooth the class probability
    word_inx = getDocWords(data, docId)
    nword_inx = getNDocWords(data, docId)
    if not useLog:
        p_class = p_class/np.sum(p_class)
        p_words = freqT[:, word_inx]
        p_nwords = freqT[:, nword_inx]
        p_nwords = 1 - p_nwords
        p_all = np.concatenate((p_words, p_nwords), axis=1)        
        for pw in p_all.T:
            p_class[0] *= pw[0]
            p_class[1] *= pw[1]    
            p_class = normalize(p_class)
       
    else:
        p_class = np.log(p_class) - np.log(p_class.sum())
        lfreqT = getLogProb(freqT)
        lfreqT2 = getLogProb(1 - freqT)
        p_words = lfreqT[:, word_inx]
        p_nwords = lfreqT2[:, nword_inx]
        p_all = np.concatenate((p_words, p_nwords), axis=1)
        #print("ALL:", p_all.T[:10])
        p_class = p_class + np.sum(p_all, axis=1)
        p_class = normalize(p_class)
    return p_class



def Classify(data, freqT, docId, lbldata, useLog):
    p_c = calcProb(data, freqT, docId, lbldata, useLog)
    if useLog:
        label = np.argmin(p_c)
    else:
        label = np.argmax(p_c)
    return label + 1

def main():
    
    wordLst = np.arange(1, getWordCount(words) + 1)
    trainDocs = getDocMatrix(trainData, trainLabel)
    testDocs = getDocMatrix(testData, testLabel)
    doclst = np.arange(1, getDocNum(trainDocs) + 1 )
    doclst2 = np.arange(1, getDocNum(testDocs) + 1 )   
    trainTotal = getDocNum(trainDocs)
    testTotal = getDocNum(testDocs)    
    wordId = 5
    #print(getLabelFreq(trainLabel))
    #print("Word:", wordId, countWordAll(wordId, trainDocs))
    #SETUP
    freqt = getFreqTable(trainDocs, trainLabel)
    rfreqt = getRelFreq(freqt, trainLabel)
    #print("RFREQ:", rfreqt)
    lfreqt = getLogProb(rfreqt)
    discrim = getDiscriminality(rfreqt, 10)
    dLst = DiscrimLst(rfreqt, wordLst)
    getTopD(dLst, wordLst, 10)
    #print(getDocWords(trainDocs, 1))
    #print("RESULT:", calcProb(trainDocs, rfreqt, 680, trainLabel))
    
    #print(getLabelName(Classify(trainDocs, rfreqt, 483, trainLabel, True)))
   
    #TEST
    resuLst = []
    resuLst2 = []
    
    for docId in np.nditer(doclst):
        resuLst.append(Classify(trainDocs, rfreqt, docId, trainLabel, True))
    
    for docId in np.nditer(doclst2):
        resuLst2.append(Classify(testDocs, rfreqt, docId, trainLabel, True))
        
    resuArr = np.asarray(resuLst)
    resuArr2 = np.asarray(resuLst2) 
    
    diff = np.where(trainLabel != resuArr)
    diff2 = np.where(testLabel != resuArr2)
    
    train_accuracy = (trainTotal - diff[0].size)/trainTotal
    test_accuracy = (testTotal - diff2[0].size)/testTotal
    
    print("Train Accuracy:", train_accuracy * 100)
    print("Test Accuracy:", test_accuracy * 100)
    
if __name__ == "__main__":
    main()