# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:55:57 2020

@author: Neeraj
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2
os.chdir('D:/Datasets/Mine/fpoints')

df = pd.read_csv('training.csv')

df.Image = df.Image.apply(lambda string: np.fromstring(string, sep=' ', dtype=np.int_))

def get(ind=0):
    temp = df.iloc[ind]
    img = temp.Image.reshape((96, 96))
    for i in range(0, len(temp[:-1]), 2):
        cv2.circle(img, tuple(temp[i:i+2].values.astype('int')), 1, (0, 255, 0), 1)
    return img

def show_sample():
    figure = plt.figure()
    for i in range(9):
        figure.add_subplot(3, 3, i+1)
        plt.xticks([])
        plt.imshow(get(i))
    plt.show()

show_sample()    
