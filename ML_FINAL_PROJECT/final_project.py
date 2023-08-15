import numpy as np
import numpy as np
import mglearn
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
################model#################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
############################################################
def readfinal(inFileName):
    # init
    recArr = []
    clsArr = []
    
    # open input text data file, format is given
    inFile = open(inFileName, 'r')
    s = inFile.readline() # skip
    
    row = 0
    while True:
        s = inFile.readline()
        data1 = s.strip() # remove leading and ending blanks
        if (len(data1) <= 0):
            break
        
        # since we use append, value must be created in the loop
        value = []
        
        strs9 = data1.split(',') # array of 31 str

        # convert to real
        for ix in range(8):
            value.append( eval(strs9[ix]) )
        # end for
        
        target = eval(strs9[8])

        recArr.append(value) ;  # add 1 record at end of array
        clsArr.append(target) ; # add 1 record at end of array
       
        row = row+1 # total read counter
    # end while
    # close input file
    inFile.close()
    # convert list to Numpy array
    npXY = np.array(recArr)
    npC  = np.array(clsArr)
    # pass out as Numpy array
    return npXY, npC
# end function

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

###############main##############################################
X,y=readfinal("C:\prog\ML\ML_FINAL_PROJECT\diabetes(after_modefied).csv")
#392比資料 2比做test 390比做5折驗證 
X_5fold, X_check, y_5fold, y_check = train_test_split(X, y, test_size = 2, random_state = 0)
Tpredict1=np.array([[2,146,70,38,360,28,0.337,29]])
Ttest1=[1]
Tpredict2=np.array([[8,186,90,35,225,34.5,0.423,37]])
Ttest2=[1]

#########################################
#5 fold data
g_numRec = 390
num_folds = 5
test_size=78
############################################


# loop through folds
total_train = 0
total_test = 0

############################knn############

neighbors_setting=range(1,11)
training_accuracy=[]
test_accuracy=[]
for num in neighbors_setting:
    total_train = 0
    total_test = 0
    for fold in range(num_folds):
        X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
        knn=KNeighborsClassifier(n_neighbors=num)
        knn.fit(X_train,y_train)
        y_predict=knn.predict(X_test)
        train_s=knn.score(X_train, y_train)
        test_s=knn.score(X_test,y_test)
        total_train += train_s
        total_test  += test_s
    average_train = total_train/num_folds
    average_test  = total_test/num_folds
    training_accuracy.append(average_train)
    test_accuracy.append(average_test)
    
#####################knn pic###################################### 
plt.title("knn model comprasion")
plt.plot(neighbors_setting,training_accuracy,label="training score")
plt.plot(neighbors_setting,test_accuracy,label="test score")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn model比較')
plt.show()

knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train,y_train)
y_predict=knn.predict(X_test)
print("knn when(n=8) training score/test score:{:.2f}/{:.2f}".format(training_accuracy[7],test_accuracy[7]))
#print(knn.score(X_check,y_check))


##############################LogisticRegression###################

from sklearn.linear_model import LogisticRegression
# loop through folds
total_train = 0
total_test = 0
############################################
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
#c=1
    logreg=LogisticRegression(C=1,max_iter=1000).fit(X_train, y_train)
    train_s=logreg.score(X_train, y_train)
    test_s=logreg.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("logic regression for c=1 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
total_train = 0
total_test = 0
logreg1=LogisticRegression(C=1,max_iter=1000).fit(X_5fold, y_5fold)
p1=logreg1.predict(Tpredict1)
p2=logreg1.predict(Tpredict2)


############################################

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
#c=100
    logreg100=LogisticRegression(C=100,max_iter=1000).fit(X_train, y_train)
    train_s=logreg.score(X_train, y_train)
    test_s=logreg.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("logic regression for c=100 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
total_train = 0
total_test = 0
############################################
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
#c=0.01
    logreg001=LogisticRegression(C=0.01,max_iter=1000).fit(X_train, y_train)
    train_s=logreg.score(X_train, y_train)
    test_s=logreg.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("logic regression for c=0.01 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
total_train = 0
total_test = 0

########logicregression pic############
feature=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabeticFunction","Age"]
plt.figure(figsize=(8,8))
plt.title("LogisticRegression(c=1) relationship picture")
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(8), feature, rotation=35)
plt.hlines(0, 0, 9)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.savefig('Logicregression(c=1)正則化後關係圖')
plt.show()

#################decision tree######################################
from sklearn.tree import DecisionTreeClassifier
# loop through folds
total_train = 0
total_test = 0
tree=DecisionTreeClassifier(random_state=0)
for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    tree=DecisionTreeClassifier(random_state=0)
    tree.fit(X_train,y_train)
    train_s=tree.score(X_train, y_train)
    test_s=tree.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("decision tree train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
#######################################################################
#max_depth=3
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    tree=DecisionTreeClassifier(max_depth=3,random_state=0)
    tree.fit(X_train,y_train)
    train_s=tree.score(X_train, y_train)
    test_s=tree.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("decision tree max_depth=3 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

#######################################################################

#max_depth=4
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    tree=DecisionTreeClassifier(max_depth=4,random_state=0)
    tree.fit(X_train,y_train)
    train_s=tree.score(X_train, y_train)
    test_s=tree.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("decision tree max_depth=4 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

#######################################################################
#max_depth=5
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    tree=DecisionTreeClassifier(max_depth=5,random_state=0)
    tree.fit(X_train,y_train)
    train_s=tree.score(X_train, y_train)
    test_s=tree.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("decision tree max_depth=5 train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

###########plot decision tree importance pic##########################
feature=["Pregnancies","Glucose","BloodPressure","SkinThickness",\
         "Insulin","BMI","DiabeticFunction","Age"]
n_feature=8
plt.title("decision tree importance picture")
plt.barh(range(8),tree.feature_importances_)
plt.yticks(range(8),feature,rotation=60)
plt.xlabel("feature importance")
plt.ylabel("feature")
plt.savefig('decision tree的importance圖')
plt.show()

##################test########################
tree3=DecisionTreeClassifier(max_depth=3,random_state=0)
tree3.fit(X_5fold, y_5fold)
p1=tree3.predict(Tpredict1)
p2=tree3.predict(Tpredict2)

##############random forest###################
from sklearn.ensemble import RandomForestClassifier
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf=RandomForestClassifier(n_estimators=100,random_state=0)
    rf.fit(X_train,y_train)
    train_s=rf.score(X_train, y_train)
    test_s=rf.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
###########################調整##########################
#######################################################################
#max_depth=3
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf3=RandomForestClassifier(max_depth=3,n_estimators=100,random_state=0)
    rf3.fit(X_train,y_train)
    train_s=rf.score(X_train, y_train)
    test_s=rf3.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree(max_depth=3) train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

#######################################################################

#max_depth=4
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf4=RandomForestClassifier(max_depth=4,n_estimators=100,random_state=0)
    rf4.fit(X_train,y_train)
    train_s=rf4.score(X_train, y_train)
    test_s=rf4.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree(max_depth=4) tree train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

#######################################################################
#max_depth=5
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf5=RandomForestClassifier(max_depth=5,n_estimators=100,random_state=0)
    rf5.fit(X_train,y_train)
    train_s=rf5.score(X_train, y_train)
    test_s=rf5.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree(max_depth=5) train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
#######################################################################
#max_depth=6
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf6=RandomForestClassifier(max_depth=6,n_estimators=100,random_state=0)
    rf6.fit(X_train,y_train)
    train_s=rf6.score(X_train, y_train)
    test_s=rf6.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree(max_depth=6) train/test score :{:.3f}/{:.3f}".format(average_train,average_test))

#######################################################################
#max_depth=7
# loop through folds
total_train = 0
total_test = 0

for fold in range(num_folds):
    X_train, X_test, y_train, y_test = TrainTestSplit_Fold(X_5fold, y_5fold, fold, test_size)
    rf5=RandomForestClassifier(max_depth=7,n_estimators=100,random_state=0)
    rf5.fit(X_train,y_train)
    train_s=rf5.score(X_train, y_train)
    test_s=rf5.score(X_test,y_test)
    total_train += train_s
    total_test  += test_s
average_train = total_train/num_folds
average_test  = total_test/num_folds
print("ramdom forest tree(max_depth=7) train/test score :{:.3f}/{:.3f}".format(average_train,average_test))
###############################################################
##################test########################
rf4=RandomForestClassifier(max_depth=4,n_estimators=100,random_state=0)
rf4.fit(X_5fold, y_5fold)
p1=rf4.predict(Tpredict1)
p2=rf4.predict(Tpredict2)

###########plot 隨機森林 importance ##########################
feature=["Pregnancies","Glucose","BloodPressure","SkinThickness",\
         "Insulin","BMI","DiabeticFunction","Age"]
n_feature=8
plt.title("random forest tree importance picture")
plt.barh(range(8),rf4.feature_importances_)
plt.yticks(range(8),feature,rotation=60)
plt.xlabel("feature importance")
plt.ylabel("feature")
plt.savefig('random forest tree的importance圖')
plt.show()

############預測自己########################
Tpredict3=np.array([[0,83,82,30.1,100,25.7,0,19]])
Tpredict4=np.array([[0,90,69,30,100,23.8,0,19]])
##############################################
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(X_5fold, y_5fold)
knnp3=knn.predict(Tpredict3)
knnp4=knn.predict(Tpredict4)
print("用knn model 預測結果{},{}".format(knnp3,knnp4))
logreg1=LogisticRegression(C=1,max_iter=1000).fit(X_5fold, y_5fold)
logp3=logreg1.predict(Tpredict3)
logp4=logreg1.predict(Tpredict4)
print("用logregression 預測結果{},{}".format(logp3,logp4))
tree3=DecisionTreeClassifier(max_depth=3,random_state=0)
tree3.fit(X_5fold, y_5fold)
treep3=tree3.predict(Tpredict3)
treep4=tree3.predict(Tpredict4)
print("用DecisionTree 預測結果{},{}".format(treep3,treep4))

rf4=RandomForestClassifier(max_depth=4,n_estimators=100,random_state=0)
rf4.fit(X_5fold, y_5fold)
rfp3=rf4.predict(Tpredict3)
rfp4=rf4.predict(Tpredict4)
print("用random forest tree 預測結果{},{}".format(rfp3,rfp4))


