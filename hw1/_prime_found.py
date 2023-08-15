# 建立
file = "C:\\prog\\ML\\hw1\\顏彥臣_prime_found.txt"
outfile = open(file, 'w')
num1 = 0
row = 1
col = 0
count = 0
for i in range(9900, 199, -1):
    for j in range(2, i):
        if not i % j and i != j:
            break
    else:
        num = i
        if row < 6:
            outfile.write(str(i))
            outfile.write(" ")
            count = count+1
            row = row+1
        elif (row == 6):
            outfile.write(str(i))
            outfile.write("\n")
            count = count+1
            row = 1
outfile.close()
print(str(count)+" primes between 200 and 9900")
num = 0

fp = open(file, 'r')
s = fp.read()
o = s.split()
print(o)
p = list(int(i)for i in o)

for num in (p):
    if num > 3000 and num < 6000:
        num1 = num1+1
print("I, 顏彥臣, 109611099, found "+str(num1) +
      " number prime numbers between 3000 and 6000")
"""
for line in fp:
    number_list.extend([int(i) for i in line.split(",")])
    print(number_list[0])
"""
"""
data = fp.readlines()
 for item in data:
 if int(item) > 3000 and int(item) < 6000:
 num = num+1
 list = item.split(",")  # 共六行
 print(a)
 for item in range(len(a)):
  if item > 3000 and item < 6000:
      num = num+1
"""
