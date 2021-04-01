import numpy as np
import cv2
import time
import os

# data = np.load('dataset_depth.npy', allow_pickle=True)

# for point in data:
#     img = point[0]
#     choice = point[1]
#     crop_img = cv2.resize(img, (480, 480))
#     cv2.imshow("image", crop_img)
#     print(img.shape)
#     if choice == 1:
#         print("yes")
#     else:
#         print("no")
#     cv2.waitKey(1)
#     time.sleep(1)


for img in os.listdir('data/train/'):
    pic = cv2.imread('data/train/' + img)
    print(pic.shape)
