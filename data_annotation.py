import numpy as np
import cv2
import os

directories = []

directory1 = 'data/bottomCamDataset/'
directory2 = 'data/streetsDataset/'

directories.append(directory1)
directories.append(directory2)

training_data = []

i = 0
for directory in directories:
    for image in os.listdir(directory):
        img = cv2.imread(directory + image)

        try:
            height, width, _ = img.shape

            new_height = 128
            new_width = 128

            upper_left = (int((width - new_width) // 2),
                          int((height - new_height) // 2))
            bottom_right = (int((width + new_width) // 2),
                            int((height + new_height) // 2))

            img = img[upper_left[1]: bottom_right[1],
                      upper_left[0]: bottom_right[0]]

            img = cv2.resize(img, (128, 128))
        except:
            print(directory + image)
            break

        output = 0
        cv2.imshow('image', img)
        cv2.moveWindow('image', 200, 200)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('y'):
            output = 1
            print('Landing Zone')
        elif key & 0xFF == ord('n'):
            output = 0
            print('Not Landing Zone')
        elif key & 0xFF == ord('d'):
            os.remove(directory + image)
            print('bad image')
            continue
        training_data.append([img, output])

        if(i % 100 == 0):
            print("Image: " + str(i))
            print("Saving to File")
            np.save("dataset_big.npy", training_data)
        i += 1

print(len(training_data))
np.save("dataset_big.npy", training_data)
