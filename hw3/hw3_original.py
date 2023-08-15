import mglearn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
#################################
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#####################################
def ReadData(inFileName):
    # init
    recArr = []
    clsArr = []

    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    s = inFile.readline()  # skip

    row = 0
    while (row < 150):
        s = inFile.readline()
        data1 = s.strip()  # remove leading and ending blanks
        if (len(data1) <= 0):
            break

        # since we use append, value must be created in the loop
        value = [0.,0.,0., 0.]

        data1 = data1.replace('[', '')  # remove [
        data1 = data1.replace(']', '')  # remove ]
        strs2 = data1.split()  # array of 2 str

        # convert to real
        value[0] = eval(strs2[0])
        value[1] = eval(strs2[1])
        value[2] = eval(strs2[2])
        value[3] = eval(strs2[3])
        #print("row = {}".format(row) + ", {}\n".format(value), end='')

        recArr.append(value)  # add 1 record at ending
        row = row+1  # total read counter
    # end while

    # 讀取種類
    s = inFile.readline()  # skip
 
    s = inFile.read()
    data1 = s.strip()  # remove leading and ending blanks

    #data1 = data1.replace('[', '')  # remove [
    #data1 = data1.replace(']', '')  # remove ]
    strs150 = data1.split()  # array of 26 str
     
    for t in strs150:
        
        clsArr.append(int(t))
        # end for
    

    # close input file
    inFile.close()

    npXY = np.array(recArr)
    npC = np.array(clsArr)

    return npXY, npC

#####################################
def Readhw3_Data(inFileName):
    # init
    recArr = []
    clsArr = []

    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    row = 0
    while (row < 150):
        s = inFile.readline()
        data1 = s.strip()  # remove leading and ending blanks
        if (len(data1) <= 0):
            break

        # since we use append, value must be created in the loop
        value = [0.,0.,0., 0.,0.]

        data1 = data1.replace('[', '')  # remove [
        data1 = data1.replace(']', '')  # remove ]
        strs2 = data1.split()  # array of 2 str

        # convert to real
        value[0] = eval(strs2[0])
        value[1] = eval(strs2[1])
        value[2] = eval(strs2[2])
        value[3] = eval(strs2[3])
        value[4] = eval(strs2[4])
        #print("row = {}".format(row) + ", {}\n".format(value), end='')

        recArr.append(value)  # add 1 record at ending
        row = row+1  # total read counter
    # end while
    x=[]
    y=[]
    #x_array=np.array(x)#1~112
    #y_array=np.array(y)#113~150
    for i in range(112):
        x.append(recArr[i])
    for i in range (112,150):
        y.append(recArr[i])
    npXY = np.array(x)
    npC=np.array(y)

    return npXY,npC
######################################
def random_store_csv(FileName,datain):
    random.shuffle(datain)
    fout = open(FileName, 'w')
    for i in data:
        fout.write("{}\n".format(i))
    fout.close()
    
######################################
############main#################
datapath="C:\\prog\\ML\\hw3\\iris_dataset.txt"
csvpath_store="C:\\prog\\ML\\hw3\\109611099_顏彥臣_iris_data.csv"
data=[]
Feature,Type=ReadData(datapath)
data=Feature.tolist()
for i in range(150):
    data[i].append(Type[i])
################隨機存取################
random_store_csv(csvpath_store,data)
################hw3#####################
csvpath_use="C:\\prog\\ML\\hw3\\109611099_顏彥臣_iris_data.csv"
X,y=Readhw3_Data(csvpath_use)
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


