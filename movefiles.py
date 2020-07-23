# _*_ coding:utf-8 _*_
import os
import shutil


f = open("./lfw_raw/imgMoreThanOne.txt")

line = f.readline()

list = os.listdir("./lfw_raw/")
num = 0
while line:
    for i in range(0, len(list)):
        fileName = os.path.basename(list[i])

        array = line.split()
        if (len(array) < 1): break

        if (fileName == array[0]):
            oldname= "./lfw_raw/"+fileName
            newname="./lfw_moreone/"+fileName
            shutil.move(oldname, newname)
            line = f.readline()
            num = num + 1

        if (i % 500 == 0): print(i)
    line = f.readline()

print ("共移动%d个文件夹"%num)
f.close()