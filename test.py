import numpy as np
import cv2
import time

data = np.load('dataset_15Fov.npy', allow_pickle=True)

for point in data:
    img = point[0]
    choice = point[1]
    crop_img = cv2.resize(img, (480, 480))
    cv2.imshow("image", crop_img)
    print(img.shape)
    if choice == 1:
        print("yes")
    else:
        print("no")
    cv2.waitKey(1)
    time.sleep(1)
