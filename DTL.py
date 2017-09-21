import numpy as np
import math
import time
#from scipy import stats
from enum import IntEnum

class Labels (IntEnum):
    atheism = 1
    graphics = 2
 
class Node:
    def __init__(self, wordId):
        self.yChild = None
        self.nChild = None
        self.wordId = wordId
        self.isRoot = False

    def isRoot(self):
        return self.isRoot
    
    def getAttr(self):
        return self.wordId
    
class DTree:
    def __init__(self, rootval, IG):
        rootNode = Node(rootval)
        rootNode.isRoot = True
        self.root = rootNode
        self.yBranch = None
        self.nBranch = None
        self.info = IG
        self.queue = []
        
    def getRoot(self):
        return self.root.isRoot
    
    def getVal(self):
        if not self.isLeaf():
            return self.root.getAttr()
        else:
            return getLabel(self.root.getAttr())
    
    def isLeaf(self):
        return self.yBranch is None and self.nBranch is None
    
    def addBranch(self, v, subtree):
        if v == 0:
            self.nBranch = subtree
        else:
            self.yBranch = subtree
            
    def Classify(self, data, docId):
        if self.isLeaf():
            resu = self.getVal()
            #print("Document", docId, "Classified as:", resu)
            return resu
        else:
            if wordInDoc(data, self.getVal(), docId):
                #print(str(getWord(self.root.getAttr()))+"? Yes")
                return self.yBranch.Classify(data, docId)
            else:
                #print(str(getWord(self.root.getAttr()))+"? No")
                return self.nBranch.Classify(data, docId)
    
    def Print(self, offset, base, filename):
        padding = offset * " "
        if isinstance(filename, str):
            writetof = True
            f = open(filename, 'a')
            
        if self.yBranch is not None  or self.nBranch is not None:
            attrlbl = padding + str(getWord(self.root.getAttr()))+"?"
            
            print(attrlbl, "IG:", self.info)
            
            extrapad = int(offset/base) * offset * " "
            print(padding + extrapad, "Yes->", end ='')
            
            if self.yBranch is not None:
                self.yBranch.Print(offset + base, base, filename)
                
            print(padding + extrapad ,"No-> ", end ='')
            if self.nBranch is not None:
                self.nBranch.Print(offset + base, base, filename)            
        else:
            print(padding, getLabelName(self.root.getAttr()))
        
#load all the data into numpy arrays    
trainData = np.loadtxt("trainData.txt")
testData = np.loadtxt("testData.txt")
trainLabel = np.loadtxt("trainLabel.txt")
testLabel = np.loadtxt("testLabel.txt")
file = open("words.txt")
words = file.read().splitlines()
labels = ["alt.atheism", "comp.graphics"]

def getWord(wordId):

    return words[wordId - 1]

def getDocWordCount(docId, data):
    return np.count_nonzero(data[docId - 1])

def getWordCount(wordLst):
    wordArr = np.zeros(len(wordLst))
    return wordArr.size

def getLabelName(labelId):
    return labels[labelId - 1]

#gets the label associated with a document
def getDocLabel(docId, lbldat):
    return lbldat[docId - 1]

#returns all the words in document: docId    
def getDoc(data,docId):
    condition = data[:,0] == docId
    f = data[condition]
    f = f.astype(int)
    if f.ndim > 1:
        return f[:,1]
    else:
        return f

def getAllDocs(data):
    f = data[:,0]
    f = f.astype(int)
    f =  np.unique(f)
    return f

def getDocMatrix(data, labels):
    data = data.astype(int)
    docLst = getAllDocs(data)
    total = labels.size
    wtotal = getWordCount(words)
    print("Total Docs:", total)
    f = np.zeros((total, wtotal))
    for dataset in data:
        #print(dataset)
        x = dataset[0] - 1
        y = dataset[1] - 1
        f[x, y] += 1
    
    return f
        
#check if word is in doc
def wordInDoc(docdata, wordId, docId):
    return docdata[docId - 1, wordId - 1]

def getLabel(lbl):
    if lbl == 1:
        return Labels.atheism
    else:
        return Labels.graphics
    
#returns the number of docs with label lbl
def getDocClasses(lbldata, lbl):
    docs = np.where(lbldata == lbl)
    return docs[0].size

#creates a probability list to calculate I and IG
def setProbLst(lbldata):
    p = getDocClasses(lbldata, Labels.atheism)
    n = getDocClasses(lbldata, Labels.graphics)
    resu = np.zeros(2)
    resu[0] = p/(p+n)
    resu[1] = n/(p+n)
    return resu

#returns the number of documents in a dataset
def getDocNum(data):
    n = data[:,0]
    return n.size
#counts the number of docs that have/doesn't have the word wordId with label: lbl

def countDocswithWord(lbldata, docdata, examples, wordId, hasWord):
    ptotal = 0
    ntotal = 0
    #returns all the documents in class lbl
    t0 = time.clock() 
    #return all the labels of the docs in examples
    flabels = lbldata[examples - 1]
    docs = np.where(flabels == Labels.atheism)
    docs2 = np.where(flabels == Labels.graphics)
    docArr = examples[docs[0]]
    docArr2 = examples[docs2[0]]
    pLst = docdata[docArr - 1, wordId - 1]
    nLst = docdata[docArr2 - 1, wordId - 1]
    if hasWord:
        ptotal = np.sum(pLst)
    else:
        ptotal = pLst.size - np.count_nonzero(pLst)
        
    if hasWord:
        ntotal = np.sum(nLst) 
    else:
        ntotal = nLst.size - np.count_nonzero(nLst)
        
    resu = np.zeros(2)
    resu[0] = ptotal
    resu[1] = ntotal
    t1 = time.clock() 
    return resu

def calcUncertainty(p):
    #print("P(V)", p)
    if np.count_nonzero(p) < p.size:
        return 0 * p
    resu = -1* p * np.log2(p)
    #print("resu", resu)
    return resu

def calcUncertainty2(p):
    #print("P(V)", p)
    if p == 0:
        return 0
    resu = -1* p * np.log2(p)
    return resu

def getEntropy(probLst): 
    pvLst = np.apply_along_axis(calcUncertainty, 0, probLst)
    return np.sum(pvLst)

def InfoGain(lbldata, probLst, docdata, examples, attr):
    return getEntropy(probLst) - getRemainder(lbldata, docdata, attr, examples)

def getRemainder(lbldata, docdata, attr, examples):
    yarr = countDocswithWord(lbldata, docdata, examples, attr, 1)
    narr = countDocswithWord(lbldata, docdata, examples, attr, 0)
    #print("Yes:" , yarr)
    #print("No:," , narr)
    total = examples.size
    sumyr = np.sum(yarr)
    sumnr = np.sum(narr)
    #print("Doesn't have word:", sumnr)
    #print("Total:" ,sumyr + sumnr)
    return sumyr/total*getEntropy(yarr/total) + sumnr/total * getEntropy(narr/total)

def chooseAttribute(lbldata, docdata, examples, attrLst):
    maxAttr = 1
    maxIG = 0
    flabels = lbldata[examples - 1]
    probLst = setProbLst(flabels)
    #print("problst", probLst)
    for wordId in attrLst:
        #IG = wordId
        IG = InfoGain(lbldata, probLst, docdata, examples, wordId)
        #print("Word:", wordId, getWord(wordId))
        if IG > maxIG:
            maxIG = IG
            maxAttr = wordId
    
    #print("Max Word \"", getWord(maxAttr), "\" with IG:", maxIG)
    resu = np.array([maxAttr, maxIG])
    return resu

def printWords(attr):
    for word in attr:
        print(getWord(word))            
    return 0
def getMode(lst):
    p = lst[lst == Labels.atheism].size
    n = lst[lst == Labels.graphics].size
    maxfreq = max(p,n)
    if maxfreq == p:
        return Labels.atheism
    else:
        return Labels.graphics
    
#returns a decision tree
#Params
#examples, a list of docs
def DTL(examples, labels, attributes, data, default, maxdepth):
    #get the classifications of each example
    classes = labels[examples - 1]
    classes = classes.astype(int)
    vlst = np.array([0,1])
    if examples.size == 0:
        #print("ending on default:",default)
        return DTree(default)
    
    elif np.unique(classes).size == 1:
        #print("ending on remaining class:",classes[0])
        return DTree(classes[0], 0)
       
    elif attributes.size == 0 or maxdepth == 0:
        mode = getMode(classes)
        print("Ran out of words or max depth reached. Using MODE:", mode)
        return DTree(mode, 0)
     
    else: 
        #filter the labels based on the example set
        bestresu = chooseAttribute(labels, data, examples, attributes)
        best = int(bestresu[0])
        maxIG = np.around(bestresu[1], decimals=3)
        tree = DTree(best, maxIG)
        attr = attributes[attributes != best]
        #print("Attr Size:", attr.size)
        mode = getMode(classes)
        #print("Curr Mode", mode)
    for x in vlst:
        eg_i = np.where(data[examples - 1, best - 1] == x)
        eg_i = examples[eg_i[0]]
        
        #perform a sanity check
        sanity = data[eg_i[0] - 1, best - 1] == x
        if not sanity:
            print("docId", eg_i[0])
            print("wordId", best, getWord(best))
            print("Actual:", data[eg_i[0] - 1, best - 1], "Expected", x)
            
        subtree = DTL(eg_i, labels, attr, data, getMode(classes), maxdepth - 1)
        tree.addBranch(x, subtree)
        
    return tree

def main():
    #SETUP
    wordLst = np.arange(1, getWordCount(words) + 1)
  
    trainDocs = getDocMatrix(trainData, trainLabel)
    testDocs = getDocMatrix(testData, testLabel)
    
    doclst = np.arange(1, getDocNum(trainDocs) + 1 )
    doclst2 = np.arange(1, getDocNum(testDocs) + 1 )
    
    trainTotal = getDocNum(trainDocs)
    testTotal = getDocNum(testDocs)
    
    #TESTS  
    maxdepth = 8
    
    classes = np.apply_along_axis(getDocLabel, 0, doclst, trainLabel)
    
    myTree = DTL(doclst, trainLabel, wordLst, trainDocs, Labels.atheism, maxdepth)
    
    #Classify the Docs
    resuLst = []
    resuLst2 = []
    for docId in np.nditer(doclst):
        resuLst.append( myTree.Classify(trainDocs, docId))
    
    for docId in np.nditer(doclst2):
        resuLst2.append( myTree.Classify(testDocs, docId))
        
    resuArr = np.asarray(resuLst)
    resuArr2 = np.asarray(resuLst2)
    diff = np.where(trainLabel != resuArr)
    diff2 = np.where(testLabel != resuArr2)
    train_accuracy = (trainTotal - diff[0].size)/trainTotal
    test_accuracy = (testTotal - diff2[0].size)/testTotal
    resustr = "Train Accuracy:" + str(train_accuracy * 100) + "; Test Accuracy" +  str(test_accuracy * 100) + ";Depth: " + str(maxdepth) + '\n'
    print(resustr)
    
if __name__ == "__main__":
    main()