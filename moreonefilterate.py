import os

f = open("./lfw_raw/nameAndNum.txt")
newTxt = "./lfw_raw/imgMoreThanOne.txt"
newf = open(newTxt, "a+")

lines = f.readlines()
print (len(lines))
num = 1
newNum = 0
for line in lines:
    array = line.split()
    if (int(array[1]) > 1):
        new_context = array[0] + '   ' + array[1] + '\n'
        newf.write(new_context)
        newNum = newNum + 1
    num = num+1
    if (num % 1000 == 0): print("%d / %d"%(num, len(lines)))

print ("There are %d lines in %s" % (newNum, newTxt))

f.close()
newf.close()