import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

dataset = np.load('dataset_depth.npy', allow_pickle=True)

images = []
labels = []

for data in dataset:
    img = data[0]
    label = data[1]

    labels.append(label)
    images.append(img)


train_imgs, test_imgs, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.25, random_state=1)

i = 0
for img in train_imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("data/train/img_"+str(i)+".png", img)
    i += 1

i = 0
for img in test_imgs:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("data/val/img_"+str(i)+".png", img)
    i += 1

test_labels = list(map(str, test_labels))

train_labels = list(map(str, train_labels))

with open('data/val/val.labels', 'w') as f:
    for label in test_labels:
        f.write(label + '\n')

with open('data/train/train.labels', 'w') as f:
    for label in train_labels:
        f.write(label + '\n')
