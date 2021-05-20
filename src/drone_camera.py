import airsim
import os
import cv2
import numpy as np
import time

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


# Async methods returns Future. Call join() to wait for task to complete.
print("TAKING OFF")
client.takeoffAsync().join()
time.sleep(1)
print("TAKEOFF COMPLETE")
client.moveToPositionAsync(0, 0, -15, 5).join()
client.moveToPositionAsync(30, 30, -15, 2)

while True:
    img = airsim.string_to_uint8_array(
        client.simGetImage("front_center", airsim.ImageType.DisparityNormalized))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    # cv2.imwrite("sample_img.png", img)
    cv2.imshow("img", img)

    # height, width, _ = img.shape

    # new_height = 128
    # new_width = 128

    # upper_left = (int((width - new_width) // 2),
    #               int((height - new_height) // 2))
    # bottom_right = (int((width + new_width) // 2),
    #                 int((height + new_height) // 2))

    # crop_img = img[upper_left[1]: bottom_right[1],
    #                upper_left[0]: bottom_right[0]].copy()

    # crop_img = cv2.resize(crop_img, (480, 480))
    # cv2.imshow("crop", crop_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        client.armDisarm(False)
        client.enableApiControl(False)
        break
