import os
import pandas as pd
import cv2

df = pd.read_csv("annotations.csv")

def train_test_csv(images, data):
    column_name = ['filename', 'width', 'height','class', 'xmin', 'ymin', 'xmax', 'ymax']   
    whole_cvs_list=[]
    for img in images:
        value_list=[]
        height, width = cv2.imread("../../workspace/training_demo/images/"+data+"/"+img,0).shape
        for val in df[df["Filename"]==img].values.tolist():
            value_list.append([val[0], width, height, "product", val[1], val[2], val[3], val[4]])
        whole_cvs_list.extend(value_list)
    csv_df = pd.DataFrame(whole_cvs_list, columns=column_name)
    csv_df.to_csv("../../workspace/training_demo/annotations/"+data+"_labels.csv")
    

if __name__ == '__main__':
    train_images = os.listdir("../../workspace/training_demo/images/train")
    train_test_csv(train_images, "train")
    print("train data done")
    test_images = os.listdir("../../workspace/training_demo/images/test")
    train_test_csv(test_images, "test")
    print("test data done")
