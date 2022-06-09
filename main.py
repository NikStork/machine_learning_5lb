# Выполнил: Филоненко Никита УВП-211

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


DIR = r'C:\Users\79057\PycharmProjects\pythonProject\RicknMorty'
CATEGORIES = ['Rick', 'Morty']

data = []

for category in CATEGORIES:
  path = os.path.join(DIR, category)
  for img in os.listdir(path):
    img_path = os.path.join(path, img)
    label = CATEGORIES.index(category)
    print(img_path)
    arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
      print('Wrong path:', path)
    else:
      new_arr = cv2.resize(arr, (60, 60))
      data.append([new_arr, label])

import random
random.shuffle(data)

X = []
Y = []

for features, label in data:
  X.append(features)
  Y.append(label)

X = np.array(X)
Y = np.array(Y)

import pickle

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('y.pkl', 'wb'))