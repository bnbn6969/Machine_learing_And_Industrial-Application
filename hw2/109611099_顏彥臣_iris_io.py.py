import numpy as np
import matplotlib.pyplot as plt
import random


fo = open("C:\\prog\\ML\\hw2\\iris_dataset.txt", 'r')
line_useless = fo.readline()  # all inputs:

# ###############3處理4個元素並存取#######################
row = 0
value = []
while True:
    line0 = fo.readline()
    line0 = line0.strip(" [")
    line0 = line0.strip('\n]')
    str4 = line0.split()
    for i in range(0, 4):
        value.append(eval(str4[i]))
    row = row+1
    if (row == 150):
        break
# 放入二維陣列 data處存
column = row
index = 0
data0 = [[0.0]*4 for i in range(column)]
data = np.array(data0)

while index < 600:
    for i in range(column):
        for j in range(4):
            data[i][j] = float(value[index])
            index = index+1

# ########################################算中位數#########################
median = []
j = 0
for j in range(4):
    arr = []
    for i in range(column):
        arr.append(data[i][j])
    median.append(np.median(arr))

# 讀
line_useless = fo.readline()  # all target:
# 讀取並計算spice#########################
data2 = []  # data2放spice
species = []
while True:
    line = fo.readline()
    if not line:
        break
    str = line.split()
    for i in range(len(str)):
        data2.append(eval(str[i]))
species.append(data2.count(0))
species.append(data2.count(1))
species.append(data2.count(2))
fo.close()

####印出################################
print('species counts:{}'.format(species))  # .format印出包括[]
for i in range(4):
    print('feature{}:median={:.2f} '.format(i, median[i]))
print("sentosa(0):{}".format(data2.count(0)))
print('versicolor (1):{}'.format(data2.count(1)))
print('virginica (2):{}'.format(data2.count(2)))

######plot###################################
xp = np.arange(0, 4, 1)  # [0 1 2 3] x軸刻度
yv = np.array(median)  # y軸刻度
plt.bar(xp, yv, width=0.5, label='Feature\'s Median',
        color='green', tick_label=xp)
plt.legend()
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

####5th加入元素##################################
sp = data2
datanew = data.tolist()
data_str = []
for i in range(150):
    datanew[i].append(data2[i])
random.shuffle(datanew)
################open file並存取##################
fout = open("109611099_顏彥臣_iris_data.csv", 'w')
for i in datanew:
    fout.write("{}\n".format(i))
fout.close()

# 計算中位數
# 1.第一個feature
'''
median = []
for j in range(4):
    num = []
    for i in range(150):
        num.append(float(alldata[i][j]))
        median.append(np.median(num))
print("{:.2f}".format(median[0]))
print("{:.2f}".format(median[1]))
print("{:.2f}".format(median[2]))
print("{:.2f}".format(median[3]))

array0 = np.array(num)
np.round([float(i) for i in array0], 2)
median_0 = np.median(array0)
print(median_0)
'''
