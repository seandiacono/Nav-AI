from operator import truediv
import sys
sys.path.append(
    'C:/Users/seand/OneDrive/Documents/University/Autonomous Drone Navigation/Implementation/AirSimAPI/packages')
import airsim
import time
import numpy as np
import cv2
import os
from scipy.spatial import distance
from random import randint, choice
from MonoDepth2.depth_predicter import DepthFinder


depth_finder = DepthFinder("mono+stereo_640x192")


z = -10

velocity = 6


def evasive_manouver(depth_img):
    global z
    height, width = depth_img.shape

    bottom_left = depth_img[(height // 2):height, 0:(width // 2)].copy()
    bottom_right = depth_img[(height // 2):height,
                             (width // 2):width].copy()
    top_left = depth_img[0:(height // 2), 0:(width // 2)].copy()
    top_right = depth_img[0:(height // 2), (width // 2):width].copy()

    top_left_avg = np.average(top_left)
    top_right_avg = np.average(top_right)
    bottom_left_avg = np.average(bottom_left)
    bottom_right_avg = np.average(bottom_right)

    if top_left_avg <= min(top_right_avg, bottom_left_avg, bottom_right_avg):
        # print("GOING TOP LEFT")
        z -= 1
        client.moveByVelocityBodyFrameAsync(0, -1, -1, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    elif top_right_avg <= min(top_left_avg, bottom_left_avg, bottom_right_avg):
        # print("GOING TOP RIGHT")
        z -= 1
        client.moveByVelocityBodyFrameAsync(0, 1, -1, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    elif bottom_right_avg <= min(top_left_avg, bottom_left_avg, top_right_avg):
        # print("GOING BOTTOM RIGHT")
        z += 1
        client.moveByVelocityBodyFrameAsync(0, 1, 1, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
    elif bottom_left_avg <= min(top_left_avg, bottom_right_avg, top_right_avg):
        # print("GOING BOTTOM LEFT")
        z += 1
        client.moveByVelocityBodyFrameAsync(0, -1, 1, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()


def reset_drone(client):
    client.moveByVelocityAsync(0, 0, 0, 1).join()
    client.hoverAsync().join()
    client.reset()
    time.sleep(2)
    client.enableApiControl(False)
    client.armDisarm(False)
    client.enableApiControl(True)
    client.armDisarm(True)


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

estimation_crashes = 0
# disparity_crashes = 0

avoidance_flights = 0

trip_times = []

for i in range(100):
    r = choice([(-50, -15), (15, 50)])

    x = randint(*r)

    r = choice([(-60, -15), (15, 60)])

    y = randint(*r)

    dist_to_dest = distance.euclidean(
        [0, 0, z], [x, y, z])

    had_avoidance = False
    z = -10
    for j in range(1):

        time_start = time.process_time()

        # Async methods returns Future. Call join() to wait for task to complete.
        # print("TAKING OFF")
        client.moveToPositionAsync(0, 0, z, 2).join()
        # print("TAKEOFF COMPLETE")
        client.moveToPositionAsync(x, y, z, velocity, yaw_mode=airsim.YawMode(is_rate=False,
                                                                              yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)

        while True:
            collided = client.simGetCollisionInfo().has_collided
            if collided and j == 0:
                print("Destination Coordinates were: x: " +
                      str(x) + " y: " + str(y))
                had_avoidance = True
                estimation_crashes += 1
                reset_drone(client)
                break
            elif collided and j == 1:
                print("Destination Coordinates were: x: " +
                      str(x) + " y: " + str(y))
                had_avoidance = True
                # disparity_crashes += 1
                reset_drone(client)
                break
            position = client.simGetVehiclePose().position

            current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                x, y, z])

            progress = int(
                100 - ((current_dist / dist_to_dest) * 100))

            if progress >= 85:
                collided = client.simGetCollisionInfo().has_collided
                if collided and j == 0:
                    print("Destination Coordinates were: x: " +
                          str(x) + " y: " + str(y))
                    had_avoidance = True
                    estimation_crashes += 1
                    reset_drone(client)
                    break
                time_end = time.process_time()
                elapsed_time = time_end - time_start
                trip_times.append(elapsed_time)
                reset_drone(client)
                break

            if j == 0:
                img = airsim.string_to_uint8_array(
                    client.simGetImage("front_center", airsim.ImageType.Scene))

                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

                depth_time_s = time.process_time()
                depth = depth_finder.get_depth_map(img)
                depth_time_e = time.process_time()
                print("Inference Time: " + str(depth_time_e - depth_time_s))
                normalizedImg = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX)
            else:
                img = airsim.string_to_uint8_array(
                    client.simGetImage("front_center", airsim.ImageType.DisparityNormalized))

                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

                normalizedImg = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX)

            ret, thresh = cv2.threshold(
                normalizedImg, 135, np.amax(normalizedImg), cv2.THRESH_BINARY)

            height, width = thresh.shape

            new_height = 200
            new_width = 200

            upper_left = (int((width - new_width) // 2),
                          int((height - new_height) // 2))
            bottom_right = (int((width + new_width) // 2),
                            int((height + new_height) // 2))

            crop_img = thresh[upper_left[1]: bottom_right[1],
                              upper_left[0]: bottom_right[0]].copy()

            average_depth = np.average(crop_img)

            # print(average_depth)

            if average_depth > 20:
                collided = client.simGetCollisionInfo().has_collided
                if collided and j == 0:
                    print("Destination Coordinates were: x: " +
                          str(x) + " y: " + str(y))
                    had_avoidance = True
                    estimation_crashes += 1
                    reset_drone(client)
                    break
                had_avoidance = True
                # print("TOO CLOSE TO OBJECT - STOPPING AND HOVERING")
                client.cancelLastTask()
                client.moveByVelocityAsync(0, 0, 0, 1).join()
                client.hoverAsync().join()
                # client.moveByVelocityBodyFrameAsync(-1, 0, 0, 2).join()
                # print("TAKING EVASIVE MANOUVER")
                evasive_manouver(crop_img)
                # print("done")
                client.moveToPositionAsync(
                    x, y, z, velocity)

    if had_avoidance:
        avoidance_flights += 1

    print("Completed Flight: " + str(i + 1))
    print("Estimation Crashes: " + str(estimation_crashes))
    # print("Disparity Crashes: " + str(disparity_crashes))
    print("Avoidance Flights: " + str(avoidance_flights))
    try:
        avg_time = sum(trip_times) / len(trip_times)
    except:
        avg_time = 0.0
    print("Average Completion Time: " + str(avg_time))

print("COMPLETE")
print("Estimation Crashes: " + str(estimation_crashes))
# print("Disparity Crashes: " + str(disparity_crashes))
print("Avoidance Flights: " + str(avoidance_flights))
avg_time = sum(trip_times) / len(trip_times)
print("Average Completion Time: " + str(avg_time))
print("EXITING")
