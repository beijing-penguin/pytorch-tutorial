import time
import os
filepath="F:\eclipse-workspace\pytorch-tutorial\logo"
# 循环输出休眠1秒
while True:
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath+"\\", allDir))
        print(child)
    time.sleep(3) # 休眠1秒