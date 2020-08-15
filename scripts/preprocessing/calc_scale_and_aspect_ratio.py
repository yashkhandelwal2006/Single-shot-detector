import os
import pandas as pd
import cv2

df = pd.read_csv("annotations.csv")

def train_test_csv(images, data):
    l=[]
    l1=[]
    l2=[]
    for img in images:
        height, width = cv2.imread("../../workspace/training_demo/images/"+data+"/"+img,0).shape
        l2.append(len(df[df["Filename"]==img].values.tolist()))
        for val in df[df["Filename"]==img].values.tolist():
            ht=val[4]-val[2]
            wt=val[3]-val[1]
            area = (wt)*(ht)
            l1.append(round(wt/ht,1))
            scale = area/(height*width)
            l.append(scale)
    d={}
    a=0
    while a < 101:
        d[a] = 0
        a+=20
    for i in l2:
        for j in d:
            if i<j:
                d[j]+=1
                break
    print("no. of products count for images")
    print(d)
    d={}
    a=0
    while a < 1.1:
        d[round(a,1)] = 0
        a+=0.2
    for i in l1:
        for j in d:
            if i<j:
                d[j]+=1
                break
    print("aspect ratio count for images")
    print(d)
    

if __name__ == '__main__':
    train_images = os.listdir("../../workspace/training_demo/images/train")
    train_test_csv(train_images, "train")
    print("train data done")
    test_images = os.listdir("../../workspace/training_demo/images/test")
    train_test_csv(test_images, "test")
    print("test data done")
