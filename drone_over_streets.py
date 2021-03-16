# import setup_path
import airsim

import sys
import time
import image_capture
import queue
import threading

print("""This script is designed to fly on the streets of the Neighborhood environment
and assumes the unreal position of the drone is [160, -1500, 120].""")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

print("arming the drone...")
client.armDisarm(True)

state = client.getMultirotorState()
if state.landed_state == airsim.LandedState.Landed:
    print("taking off...")
    client.takeoffAsync().join()
else:
    client.hoverAsync().join()

time.sleep(1)

state = client.getMultirotorState()
# if state.landed_state == airsim.LandedState.Landed:
#     print("take off failed...")
#     sys.exit(1)

# AirSim uses NED coordinates so negative axis is up.
# z of -7 is 7 meters above the original launch point.
z = -40
print("make sure we are hovering at 7 meters...")
client.moveToZAsync(z, 2).join()

# see https://github.com/Microsoft/AirSim/wiki/moveOnPath-demo

# this method is async and we are not waiting for the result since we are passing timeout_sec=0.

stop_thread = False
que = queue.Queue()
thread = threading.Thread(
    target=lambda q, arg1: q.put(image_capture.image_capture(arg1)), args=(que, lambda: stop_thread))
thread.start()


print("flying on path...")
result = client.moveOnPathAsync([airsim.Vector3r(125, 0, z),
                                 airsim.Vector3r(125, -130, z),
                                 airsim.Vector3r(0, -130, z),
                                 airsim.Vector3r(0, 0, z)],
                                2,
                                drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(False, 0), lookahead=20, adaptive_lookahead=1).join()


stop_thread = True
thread.join()
images = que.get()
for i in range(0, len(images)):
    airsim.write_file('data/streetsDataset/img_'+str(i)+'.png',
                      images[i])
    # except:
# drone will over-shoot so we bring it back to the start point before landing.
client.moveToPositionAsync(0, 0, z, 1).join()
print("landing...")
client.landAsync().join()
print("disarming...")
client.armDisarm(False)
client.enableApiControl(False)
print("done.")
