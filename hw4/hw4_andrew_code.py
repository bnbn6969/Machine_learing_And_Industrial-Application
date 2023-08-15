#=========================================================
# NYCU  2022FALL 516148 : BOSTON EXAMPLE - Python Homework 4
#=========================================================

import numpy as np
#import scipy
import mglearn
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import random



# use import train_test_split (sklearn)
from sklearn.model_selection import train_test_split



g_BOSTON  = "C:\\prog\\ML\\hw4\\hw4_boston.csv"



############################################################
# function to read from input CSV file
############################################################
def readBoston(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    
    row = 1
    while True:
        s = inFile.readline()
        data1 = s.strip() # remove leading and ending blanks
        if (len(data1) <= 0):
            break
        
        # since we use append, value must be created in the loop
        values = []
        
        #data1 = data1.replace('[', '') # remove [
        #data1 = data1.replace(']', '') # remove ]
        
        data1 = data1.replace(',', ' ') # replace comma

        # split
        strs105 = data1.split() # array of 105 str

        num = len(strs105)
        if (num != 105):
            print("==== Error at input data file, line {} ====".format(row))
            break
        
        # convert to real
        for sv in strs105:
            dv = eval(sv)
            values.append(dv)
        # end for

        #print("row = {}".format(row) + ", {}\n".format(value), end='')

        # last item is target
        # get last item value and delete last item from the list
        target = values.pop() # pop last item, so list has lost 1 item
        clsArr.append(target)

        # last item in the CSV is deleted, so we have all features
        recArr.append(values)  # add 1 record at ending
        
        row = row+1 # total read counter
    # end while
    
    # close input file
    inFile.close()

    # convert to numpy array
    npXY = np.array(recArr)
    npC  = np.array(clsArr)

    # return numpy array
    return npXY, npC
# end function




############################################################
# Given feature Numpy array and target Numpy array, return
# training and test datasets
#
# Input Arguments
# X - feature Numpy array (2-dim)
# y - target Numpy array (1-dim)
# fold - which fold, starting from 0
# test_size - int, as size of test datasets
#
# NOTE
# 1. X & y should be randomly arranged before passing in
# 2. For the fold 0, it's the first test_size items
#    For the fold 1, it's the [test_size - 2*test_size]
#    The caller must make sure that the fold # and test_size
#    will not exceed the X & y size
#
# Return
# X_train: feature array of training dataset (total - test_size)
# X_test : feature array of test dataset (test_size)
# y_train: target array of training dataset
# y_test : target array of test dataset
############################################################
def TrainTestSplit_Fold(X, y, fold, test_size):
    # safety check
    if (fold < 0):
        fold = 0
    # end if
    
    # input features
    numValue = X.size
    rows = len(X)
    cols = int(numValue/rows)
    
    # safety check
    rmns = numValue % rows
    if (rmns != 0):
        print("ERROR - missing data in X")
    # end if
    

    # safety check
    #numValue2 = y.size
    #rows2 = len(y)
    #cols2 = int(numValue/rows)
    #print("rows = {}, ".format(rows) + "column = {}".format(cols))

    # for given fold, this is the range(t0, t1)
    t0 = fold * test_size
    t1 = t0 + test_size
    # safety check
    if (t1 > rows):
        print("ERROR - out of bound")
        t1 = rows
    #end if

    # test dataset Numpy array
    fea_test = X[t0:t1,:] # doesn't include t1 item
    tar_test = y[t0:t1]   # doesn't include t1 item

    # training dataset Numpy array
    dr = [t0+x for x in range(test_size)] # test dataset items in a list
    #print(dr) # delete these test dataset items
    
    fea_train = np.delete(X, dr, 0)   # remove test dataset items
    tar_train = np.delete(y, dr, 0)   # remove test dataset items

    return fea_train, fea_test, tar_train, tar_test
# end function



#=========================================================
# LINEAR MODEL EXAMPLE - Python textbook exercise
#=========================================================
# Linear regression (aka ordinary least squares)
# Boston Housing dataset - this dataset has 506 samples and 104 derived features plus 1 target 




###########################################################
# main

#from sklearn.model_selection import train_test_split

# Ridge regression - L2 regularization
from sklearn.linear_model import Ridge

# read data from CSV file
X, y = readBoston(g_BOSTON)


# ML total records in csv data file, a given number

# 86 data for test, 420 data for training/validation
X_5fold, X_check, y_5fold, y_check = train_test_split(X, y, test_size = 86, random_state = 0)
print(X_5fold.shape)
print(X_check.shape)


# ML total records in 5-fold data set, a given number
g_numRec = 420

num_folds = 5
test_size = int(g_numRec * (1.0/num_folds)) # 101




# Ridge regression
# increase alpha to help regularization but reduce training sets performance
# Ridge default alpha 1.0
r_alpha = 1
rs = str(r_alpha)
rs = rs.strip()
rs = rs.rstrip('0')

# loop through folds
total_train = 0
total_test = 0
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    lr = Ridge(alpha = r_alpha).fit(X_train, y_train) # y_train is 1-dim array

    train_s = lr.score(X_train, y_train)
    test_s  = lr.score(X_test,  y_test)

    print("Ridge (alpha {}) Boston, fold {}, Train/Test score: {:.2f}/{:.2f}".format( \
                                    rs, fold, train_s, test_s))
    total_train += train_s
    total_test  += test_s
# end for

# average score for 5-fold training model
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("\nRidge (alpha {}) Boston, 5-fold Train/Test average score: {:.2f}/{:.2f}".format( \
                                    rs, average_train, average_test))


# use 5-fold training model to verify the reserved X_check (real test) datasets
lr5 = Ridge(alpha = r_alpha).fit(X_5fold, y_5fold) # y_5fold is 1-dim array
fold5_s = lr5.score(X_5fold,  y_5fold)
check_s = lr5.score(X_check,  y_check)
print("\nRidge (alpha {}) Boston, 5-fold/verify score: {:.2f}/{:.2f}".format( \
                                    rs, fold5_s, check_s))



























"""
# =================== Lasso regression - L1 regularization
from sklearn.linear_model import Lasso

# Lasso regression
s_alpha = 0.010
rs = str(s_alpha)
rs = rs.strip()
rs = rs.rstrip('0')

# loop through folds
print()
total_train = 0
total_test = 0
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    ls = Lasso(alpha = s_alpha, max_iter=100000).fit(X_train, y_train) # y_train is 1-dim array

    train_s = ls.score(X_train, y_train)
    test_s  = ls.score(X_test,  y_test)

    print("Lasso (alpha {}) Boston, fold {}, Train/Test score: {:.2f}/{:.2f}".format( \
                                    rs, fold, train_s, test_s))
    total_train += train_s
    total_test  += test_s
# end for

# average score for 5-fold training model
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("\nLasso (alpha {}) Boston, 5-fold Train/Test average score: {:.2f}/{:.2f}".format( \
                                    rs, average_train, average_test))


# use 5-fold training model to verify the reserved X_check (real test) datasets
ls5 = Lasso(alpha = s_alpha, max_iter=100000).fit(X_5fold, y_5fold) # y_5fold is 1-dim array
fold5_s = ls5.score(X_5fold,  y_5fold)
check_s = ls5.score(X_check,  y_check)
print("\nLasso (alpha {}) Boston, 5-fold/verify score: {:.2f}/{:.2f}".format( \
                                    rs, fold5_s, check_s))







# =================== ElasticNet regression - L1 + L2 regularization
from sklearn.linear_model import ElasticNet

# ElasticNet regression
e_alpha = 0.01
l1_ratio = 0.4
rs = str(e_alpha)
rs = rs.strip()
rs = rs.rstrip('0')
rt = str(l1_ratio)
rt = rt.strip()
rt = rt.rstrip('0')

# loop through folds
print()
total_train = 0
total_test = 0
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    els = ElasticNet(alpha = e_alpha, l1_ratio = l1_ratio, max_iter=100000).fit(X_train, y_train) # y_train is 1-dim array

    train_s = els.score(X_train, y_train)
    test_s  = els.score(X_test,  y_test)
    
    print("ElasticNet (alpha {}, L1_ratio {}) Boston, fold {}, Train/Test score: {:.2f}/{:.2f}".format( \
                                    rs, rt, fold, train_s, test_s))
    total_train += train_s
    total_test  += test_s
# end for

# average score for 5-fold training model
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("\nElasticNet (alpha {}, L1_ratio {}) Boston, 5-fold Train/Test average score: {:.2f}/{:.2f}".format( \
                                    rs, rt, average_train, average_test))


# use 5-fold training model to verify the reserved X_check (real test) datasets
els5 = ElasticNet(alpha = e_alpha, l1_ratio = l1_ratio, max_iter=100000).fit(X_5fold, y_5fold) # y_5fold is 1-dim array
fold5_s = els5.score(X_5fold,  y_5fold)
check_s = els5.score(X_check,  y_check)
print("\nElasticNet (alpha {}, L1_ratio {}) Boston, 5-fold/verify score: {:.2f}/{:.2f}".format( \
                                    rs, rt, fold5_s, check_s))
"""

