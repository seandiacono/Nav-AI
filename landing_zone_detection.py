import airsim
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import time

model = keras.models.load_model('models/LandNet_15Fov')

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
print("TAKING OFF")
client.takeoffAsync().join()
print("TAKEOFF COMPLETE")
print("FLYING TO 40 ALTITUDE")
client.moveToPositionAsync(0, 0, -40, 5).join()
print("FLYING TO DESTINATION")

x = 43
y = 43

client.moveToPositionAsync(x, y, -40, 5).join()
print("ARRIVED AT DESTINATION")

time.sleep(4)

landed = False

while not landed:
    img = airsim.string_to_uint8_array(
        client.simGetImage("bottom_center", airsim.ImageType.Scene))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (128, 128))

    img2show = cv2.resize(img, (480, 480))

    img = np.asarray([img])

    pred = model.predict_classes(img)
    pred = pred.reshape(1, -1)[0]

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

    if pred == 1:
        print("LANDING")
        client.moveToPositionAsync(x, y, -0.1, 1).join()
        landed = True
    else:
        print("NOT SAFE TO LAND...MOVING TO NEW POSITION")
        x -= 1
        client.moveToPositionAsync(x, y, -40, 0.5).join()
        print("CHECKING NEW SPOT")

client.armDisarm(False)
client.enableApiControl(False)
print("Exiting..")
cv2.waitKey(0)
cv2.destroyAllWindows()
