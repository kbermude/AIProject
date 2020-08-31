import cv2
import numpy as np 
import os
import sys
data=[]
labels=[]
from PIL import Image
ruta = './'
height = 30
width = 30
channels = 3
classes = 15#[0,1,2,3,4,5,6,7]
n_classes=classes#len(classes)
n_inputs = height * width*channels
path1 = os.path.abspath(os.path.dirname(__file__))
for i in range(classes) :
    path = ruta+"Train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print("ERROR en "+path+a)
Cells=np.array(data)
labels=np.array(labels)