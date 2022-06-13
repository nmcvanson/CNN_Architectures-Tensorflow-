import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os

def traffic_sign_dataset():
    data =[]
    labels = []
    classes = 43
    os.chdir('./datasets/traffic_signs_dataset')
    cur_path = os.getcwd()
    for i in range(classes):     
        path = os.path.join(cur_path,'Train', str(i))     
        images = os.listdir(path)
        for a in images:
            try:             
                image = Image.open(path +'\\'+ a)             
                image = image.resize((64,64)) 
                # Resizing all images into 30*30                                                 
                image =np.array(image)             
                data.append(image)             
                labels.append(i)
            except Exception as e:
                print(e)
    data = np.array(data) 
    labels = np.array(labels)
    print(data.shape, labels.shape)
    # os.mkdir('Training')
    np.save('./Training/data', data)
    np.save('./Training/target', labels)
    return data, labels

if __name__ =="__main__":
    data, labels = traffic_sign_dataset()