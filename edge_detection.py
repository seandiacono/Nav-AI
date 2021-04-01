import numpy as np
import cv2
import time

# data = np.load('dataset_1.npy', allow_pickle=True)

# for point in data:
#     img = point[0]
#     choice = point[1]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 100, 200)
#     cv2.imshow("image", edges)
#     if choice == 1:
#         print("yes")
#     else:
#         print("no")
#     cv2.waitKey(1)
#     time.sleep(1)

img = cv2.imread('testImages/013.jpg/013.jpg')
img = cv2.resize(img, (400,400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# edges = cv2.Canny(gray, 200, 120)
cv2.imshow("image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
