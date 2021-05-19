import sys
sys.path.append(
    'C:/Users/seand/OneDrive/Documents/University/Autonomous Drone Navigation/Implementation/AirSimAPI/packages')
import airsim
import time
import numpy as np
import cv2
from MonoDepth2.depth_predicter import DepthFinder


depth_finder = DepthFinder("mono+stereo_640x192")

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

x = 28
y = -10
z = -10

# Async methods returns Future. Call join() to wait for task to complete.
print("TAKING OFF")
client.moveToPositionAsync(0, 0, z, 2).join()
time.sleep(1)
print("TAKEOFF COMPLETE")
client.moveToPositionAsync(x, y, z, 2, yaw_mode=airsim.YawMode(is_rate=False,
                                                               yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)

time.sleep(6)


def evasive_manouver(depth_img):
    height, width = depth_img.shape

    left = depth_img[0:height, 0:(width // 2)].copy()
    right = depth_img[0:height, (width // 2):width].copy()

    left_avg = np.average(left)
    right_avg = np.average(right)

    if left_avg > right_avg:
        print("GOING RIGHT")
        client.moveByVelocityBodyFrameAsync(0, 1, 0, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    else:
        print("GOING LEFT")
        client.moveByVelocityBodyFrameAsync(0, -1, 0, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()


while True:
    img = airsim.string_to_uint8_array(
        client.simGetImage("front_center", airsim.ImageType.Scene))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    depth = depth_finder.get_depth_map(img)
    normalizedImg = cv2.normalize(
        depth, None, 0, 255, cv2.NORM_MINMAX)

    ret, thresh = cv2.threshold(
        normalizedImg, 175, np.amax(normalizedImg), cv2.THRESH_BINARY)

    height, width, _ = img.shape

    new_height = 200
    new_width = 200

    upper_left = (int((width - new_width) // 2),
                  int((height - new_height) // 2))
    bottom_right = (int((width + new_width) // 2),
                    int((height + new_height) // 2))

    crop_img = thresh[upper_left[1]: bottom_right[1],
                      upper_left[0]: bottom_right[0]].copy()

    cv2.imshow("cropped", crop_img)
    cv2.imshow("depth", depth)
    average_depth = np.average(crop_img)

    print(average_depth)

    if average_depth > 20:
        print("TOO CLOSE TO OBJECT - STOPPING AND HOVERING")
        # client.cancelLastTask()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
        client.hoverAsync().join()
        print("TAKING EVASIVE MANOUVER")
        evasive_manouver(crop_img)
        print("done")
        client.moveToPositionAsync(
            x, y, z, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.cancelLastTask()
        client.moveByVelocityAsync(0, 0, 0, 1)
        client.hoverAsync()
        break

print("EXITING")
