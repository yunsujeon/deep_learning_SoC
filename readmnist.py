import numpy as np
import time
from struct import *
#파일 읽기
fp_image = open('t10k-images.idx3-ubyte','rb')
fp_label = open('t10k-labels.idx1-ubyte','rb')

#사용할 변수 초기화
img = np.zeros((28,28)) #이미지가 저장될 부분
lbl = [ [],[],[],[],[],[],[],[],[],[] ] #숫자별로 저장 (0 ~ 9)
oneimg = np.zeros(784)
onearr = []

d = 0
l = 0
index=0

s = fp_image.read(16)    #read first 16byte
l = fp_label.read(8)     #read first  8byte

k=0 #테스트용 index


def divideBy256(x):
    return x / 256

def sixformat(x):
    return format(x, ".6f")
# while True:
for i in range(101):
    s = fp_image.read(784) #784바이트씩 읽음
    l = fp_label.read(1) #1바이트씩 읽음

    if not s:
        break;
    if not l:
        break;

    index = int(l[0])
    index = str(index)
    oneimg = unpack(len(s) * 'B', s)
    onearr = oneimg
    newList = list(map(divideBy256, onearr))
    newLists = list(map(sixformat, newList))
    #파일입출력으로 메모장에 저장

    img = np.reshape(unpack(len(s) * 'B', s), (28, 28))

    with open('C:/Users/dbstn/Desktop/numbers.txt', 'a') as f:
        strr = "\n"
        f.write(index)
        f.write(strr)
        f.write(', '.join(newLists))
        f.write(strr)
        f.close()

    print(i+1," 번 반복했습니다")