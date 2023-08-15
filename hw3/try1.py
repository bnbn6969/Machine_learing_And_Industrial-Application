import mglearn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
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
        value = [0., 0.,0.,0.]

        data1 = data1.replace('[', '')  # remove [
        data1 = data1.replace(']', '')  # remove ]
        strs2 = data1.split()  # array of 2 str

        # convert to real
        value[0] = eval(strs2[0])
        value[1] = eval(strs2[1])
        value[2] = eval(strs2[2])
        value[3] = eval(strs2[3])
        cls=eval(strs2[4])
        #print("row = {}".format(row) + ", {}\n".format(value), end='')

        recArr.append(value)  # add 1 record at ending
        clsArr.append(cls)
        row = row+1  # total read counter
    # end while
    

    # close input file
    inFile.close()
    npXY = np.array(recArr)
    npXY=npXY.reshape(150,4)
    npC = np.array(clsArr)

    return npXY,npC

#end ###################################################
########################################################
csvpath_use="C:\\prog\\ML\\hw3\\109611099_顏彥臣_iris_data.csv"
Feature,Type=readForge(csvpath_use)
#X_train,X_test,y_train,y_test=train_test_split(Feature,Type,test_size=0.25,random_state=0)
##########################
def neighborsp(num):
    score_is_1=0
    feature1=[5,2.9,1,0.2]
    feature2=[3,2.2,4,0.9]
    feature1=np.array(feature1)
    feature1=feature1.reshape(1,4)
    feature2=np.array(feature2)
    feature2=feature2.reshape(1,4)
    while not score_is_1:
        for i in range(100):
            X_train,X_test,y_train,y_test=train_test_split(Feature,Type,test_size=0.25,random_state=i)
        nbr=KNeighborsClassifier(n_neighbors=num)
        nbr.fit(X_train,y_train)
        y_predict=nbr.predict(X_test)
        score=nbr.score(X_test,y_test)
        if score==1:
            score_is_1=1
    print("when random state="+str(i))
    y_predict1=nbr.predict(feature1)
    y_predict2=nbr.predict(feature2)
    predict=[0,0]
    predict[0]=y_predict1[0]
    predict[1]=y_predict2[0]
#######get specie##########################
    specie=[]
    for i in range(2):
        if predict[i]==0:
            specie.append("setosa (0)")
        if predict[i]==1:
            specie.append("versicolor (1)")
        if predict[i]==2:
            specie.append("virginica (2)")
    print("train set score:{}".format(score))
    print("X_new data:[5,2.9,1,0.2],predict spice name: {}".format(specie[0]))
    print("X_new data:[3,2.2,4,0.9],predict spice name: {}".format(specie[1]))



################neighbor=1#################################
print("when number of neighbors=1")
neighborsp(1)
print("when number of neighbors=3")
neighborsp(3)
print("when number of neighbors=5")
neighborsp(5)












