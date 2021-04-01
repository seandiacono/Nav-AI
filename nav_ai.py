from __future__ import print_function
import airsim
from numpy.lib.function_base import average
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import time
from AdaBins.obstacle_avoidance import DepthFinder
import math


def navigation(x, y, z, vel):
    print('Flying to Desired Destination', end='\r')
    client.moveToPositionAsync(
        x, y, z, vel, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20).join()

    # position = client.simGetVehiclePose().position

    # actual_x = position.x_val
    # actual_y = position.y_val

    # while math.dist((actual_x, actual_y), (x, y)) > 1:
    #     position = client.simGetVehiclePose().position

    #     actual_x = position.x_val
    #     actual_y = position.y_val

    # img = airsim.string_to_uint8_array(
    #     client.simGetImage("front_center", airsim.ImageType.Scene))

    # img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    # depth = depth_finder.get_depth_map(img)
    # ret, thresh = cv2.threshold(
    #     depth, 2500, np.amax(depth), cv2.THRESH_BINARY_INV)

    # height, width = thresh.shape

    # upper_left = (width // 4, height // 4)
    # bottom_right = (width * 3 // 4, height * 3 // 4)

    # crop_img = thresh[upper_left[1]: bottom_right[1] +
    #                   1, upper_left[0]: bottom_right[0] + 1].copy()

    # height, width = crop_img.shape

    # crop_img = crop_img[height // 2:height, 0:width]
    # average_depth = np.average(crop_img)

    # if average_depth > 8500:
    #     print("TOO CLOSE TO OBJECT - STOPPING AND HOVERING",
    #           end='\r')
    #     client.cancelLastTask()
    #     client.moveByVelocityAsync(0, 0, 0, 1)
    #     client.hoverAsync().join()
    #     print("TAKING EVASIVE MANOUVER", end='\r')
    #     obstacle_avoidance(crop_img)
    #     client.moveToPositionAsync(
    #         x, y, z, vel, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)

    # cv2.imshow("crop_img", crop_img)
    # cv2.imshow("est_depth", depth)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
    print('Arrived at Desired Destination', end='\r')
    return True


def obstacle_avoidance(img):
    height, width = img.shape

    left = img[0:height, 0:(width // 2)].copy()
    right = img[0:height, (width // 2):width].copy()

    left_avg = np.average(left)
    right_avg = np.average(right)

    if left_avg > right_avg:
        print("GOING RIGHT", end='\r')
        client.moveByVelocityAsync(0, 0.5, 0, 2).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    else:
        print("GOING LEFT", end='\r')
        client.moveByVelocityAsync(0, -0.5, 0, 2).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()


def land(x, y, z):
    landed = False

    while not landed:
        img = airsim.string_to_uint8_array(
            client.simGetImage("bottom_center", airsim.ImageType.Scene))

        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        height, width, _ = img.shape

        new_height = 64
        new_width = 64

        upper_left = (int((width - new_width) // 2),
                      int((height - new_height) // 2))
        bottom_right = (int((width + new_width) // 2),
                        int((height + new_height) // 2))

        img = img[upper_left[1]: bottom_right[1],
                  upper_left[0]: bottom_right[0]]

        img = cv2.resize(img, (64, 64))

        img2show = cv2.resize(img, (480, 480))

        img = np.asarray([img])

        pred = model.predict(img)
        print(pred[0][0])

        if pred == 0:
            text = "No Landing Zone"
            color = (0, 0, 255)
        else:
            text = "Landing Zone"
            color = (0, 255, 0)

        img2show = cv2.putText(img2show, text, (15, 450),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
        cv2.imshow("img", img2show)
        cv2.waitKey(1)

        pred = 0
        if pred == 1:
            print("LANDING")
            client.moveToPositionAsync(x, y, -0.1, 1).join()
            landed = True
        else:
            print("NOT SAFE TO LAND...MOVING TO NEW POSITION")
            x -= 1
            client.moveToPositionAsync(x, y, z, 0.5).join()
            print("CHECKING NEW SPOT")


def main():
    print("\n\n***Welcome to NavAI***")
    z = -40
    print("Enter 'X' coordinate of destination")
    x = int(input())
    print("Enter 'Y' coordinate of destination")
    y = int(input())
    print("Enter Cruising Velocity (Recommended: 1)")
    vel = int(input())

    print("\nCurrent Action:")

    print('Taking Off...', end='\r')
    client.takeoffAsync().join()
    print('Takeoff Complete.', end='\r')

    print('Rising to 40m', end='\r')
    client.moveToPositionAsync(0, 0, z, 2).join()

    arrived = navigation(x, y, z, vel)

    if arrived:
        land(x, y, z)


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

depth_finder = DepthFinder(dataset='kitti')

model = keras.models.load_model('models/LandNet_BigDataset')

main()

client.armDisarm(False)
client.enableApiControl(False)
