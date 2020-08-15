import pandas as pd
import os

test_images = os.listdir("images/test")

df=pd.read_csv("annotations/test_labels.csv")

for image in test_images:
    file = open("evaluation/groundtruths/"+image.replace('.JPG','.txt').replace('.jpg','.txt'), "w") 
    text=""
    print(image)
    print(len(df[df["filename"]==image].values.tolist()))
    for val in df[df["filename"]==image].values.tolist():
        text+='product '+ ' '.join([str(i) for i in val[4:]])+'\n'
    file.write(text)
    file.close()
