import mglearn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
###################################################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


############################################################
def readForge(inFileName):
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
        value
        recArr.append(value)  # add 1 record at ending
        row = row+1  # total read counter
    # end while
    inFile.close()
    npXY = np.array(recArr)
    npXY=npXY.reshape(150,5)
   
    return npXY

#end ###################################################
#def seperate:
    
########################################################
csvpath_use="C:\\prog\\ML\\hw3\\109611099_顏彥臣_iris_data.csv"
data=readForge(csvpath_use)
random.shuffle(data)

'''
score_is_1=False
while not score_is_1:
    for i in range(100):
        X_train,X_test,y_train,y_test=train_test_split(Feature,Type,test_size=0.25,random_state=i)
    nbr1=KNeighborsClassifier(n_neighbors=1)
    nbr1.fit(X_train,y_train)
    y_predict=nbr1.predict(X_test)
    score1=nbr1.score(X_test,y_test)
    if score1==1:
        score_is_1=True
''' 
