import airsim
import time


def image_capture(stop):
    i = 0
    images = []
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    while True:
        print("Image: " + str(i))
        image = client.simGetImage(
            "bottom_center", airsim.ImageType.DepthVis)
        images.append(image)
        i = i + 1
        if stop():
            return images
        time.sleep(0.5)
