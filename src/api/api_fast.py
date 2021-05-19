import sys
sys.path.append(
    'C:/Users/seand/OneDrive/Documents/University/Autonomous Drone Navigation/Implementation/AirSimAPI/packages')
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import airsim
import numpy as np
import cv2
import base64
from scipy.spatial import distance
import torch
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import stats
from MonoDepth2.depth_predicter import DepthFinder
import threading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)


class DroneController(BaseModel):

    xCoord: float = None
    yCoord: float = None
    altitude: float = None
    velocity: float = None

    progress: Optional[int] = 0
    dist_to_dest: Optional[float] = 0.0
    arrived: Optional[bool] = False
    status: Optional[Any] = "Idle"
    time: Optional[float] = 0.0
    encoded_img: Optional[Any] = ""

    def evasive_manouver(self, depth_img):
        global client

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

    def percent_landing_zone(self, img):
        unique, counts = np.unique(
            img.reshape(-1, img.shape[2]), axis=0, return_counts=True)

        tuples = [tuple(x) for x in unique]

        total = sum(counts)

        percentages = [(x / total) * 100 for x in counts]

        dict_percentages = dict(zip(tuples, percentages))

        try:
            landing_zone_percent = dict_percentages[(128, 64, 128)]
        except:
            landing_zone_percent = 0.0

        return landing_zone_percent

    def predict_image(self, model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        model.to(device)
        image = image.to(device)
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = model(image)
        return output

    def decode_segmap(self, image, nc=23):

        label_colors = np.array([(0, 0, 0),  # 0=unlabeled
                                 # 1=paved-area, 2=dirt, 3=bird, 4=grass, 5=gravel
                                 (128, 64, 128), (130, 76, 0), (0,
                                                                102, 0), (112, 103, 87), (28, 42, 168),
                                 # 6=water, 7=rocks, 8=pool, 9=vegetation, 10=roof
                                 (48, 41, 30), (0, 50, 89), (107,
                                                             142, 35), (70, 70, 70), (102, 102, 156),
                                 # 11=wall, 12=window, 13=door, 14=fence, 15=fence-pole
                                 (254, 228, 12), (254, 148, 12), (190, 153,
                                                                  153), (153, 153, 153), (255, 22, 96),
                                 # 16=person, 17=dog, 18=car, 19=bicycle, 20=tree, 21=bald-tree, 22=ar-marker, 23=obstacle
                                 (102, 51, 0), (9, 143, 150), (119, 11, 32), (51, 51, 0), (190, 250, 190), (112, 150, 146), (2, 135, 115)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def bincount_app(self, a):
        a2D = a.reshape(-1, a.shape[-1])
        col_range = (256, 256, 256)  # generically : a2D.max(0)+1
        a1D = np.ravel_multi_index(a2D.T, col_range)
        return np.unravel_index(np.bincount(a1D).argmax(), col_range)

    def landing(self):
        global client

        print("rotating north")
        client.moveByRollPitchYawThrottleAsync(0.0, 0.0, 0.0, 0.5, 1).join()
        print("done")

        landed = False

        newX = self.xCoord
        newY = self.yCoord

        while not landed:
            img = airsim.string_to_uint8_array(
                client.simGetImage("bottom_center", airsim.ImageType.Scene))

            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (1056, 704))

            pred_mask = self.predict_image(model, img)
            pred_mask = torch.argmax(
                pred_mask.squeeze(), dim=0).detach().cpu().numpy()

            pred_mask = self.decode_segmap(pred_mask)

            height, width, _ = img.shape

            new_height = 100
            new_width = 100

            upper_left = (int((width - new_width) // 2),
                          int((height - new_height) // 2))
            bottom_right = (int((width + new_width) // 2),
                            int((height + new_height) // 2))

            img = pred_mask[upper_left[1]: bottom_right[1],
                            upper_left[0]: bottom_right[0]].copy()

            landing_zone_percent = self.percent_landing_zone(img)

            if landing_zone_percent >= 95.0:
                client.moveToPositionAsync(
                    newX, newY, 1, 2).join()
                client.landAsync()
                landed = True
            else:
                height, width, _ = pred_mask.shape

                bottom_left = pred_mask[(height // 2)
                                         :height, 0:(width // 2)].copy()
                bottom_right = pred_mask[(height // 2):height,
                                         (width // 2):width].copy()
                top_left = pred_mask[0:(height // 2), 0:(width // 2)].copy()
                top_right = pred_mask[0:(height // 2),
                                      (width // 2):width].copy()

                top_left_percent = self.percent_landing_zone(top_left)
                top_right_percent = self.percent_landing_zone(top_right)
                bottom_left_percent = self.percent_landing_zone(bottom_left)
                bottom_right_percent = self.percent_landing_zone(bottom_right)

                if top_left_percent > max(top_right_percent, bottom_left_percent, bottom_right_percent):
                    print("Moving Top left")
                    newX += 2
                    newY += -2
                elif top_right_percent > max(top_left_percent, bottom_left_percent, bottom_right_percent):
                    print("Moving Top right")
                    newX += 2
                    newY += 2
                elif bottom_left_percent > max(top_left_percent, top_right_percent, bottom_right_percent):
                    print("Moving bottom left")
                    newX += -2
                    newY += -2
                elif bottom_right_percent > max(top_left_percent, top_right_percent, bottom_left_percent):
                    print("Moving bottom right")
                    newX += -2
                    newY += 2

                client.moveToPositionAsync(
                    newX, newY, self.altitude, 1).join()
                client.moveByVelocityAsync(0, 0, 0, 1).join()

        return

    def get_status(self):
        return {"progress": self.progress, "time_left": int(self.time), "status": self.status, "image": self.encoded_img}

    def navigate(self):

        self.status = "Initialising"

        global client

        self.arrived = False

        client.armDisarm(True)
        client.takeoffAsync()

        # rise to altitude
        client.enableApiControl(True)
        client.moveToPositionAsync(0, 0, self.altitude, 2).join()

        self.dist_to_dest = distance.euclidean(
            [0, 0, self.altitude], [self.xCoord, self.yCoord, self.altitude])

        client.moveToPositionAsync(
            self.xCoord, self.yCoord, self.altitude, self.velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)

        CAMERA_NAME = '0'
        IMAGE_TYPE = airsim.ImageType.Scene
        DECODE_EXTENSION = '.jpeg'

        self.status = "Flying"

        while not self.arrived:
            position = client.simGetVehiclePose().position

            current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                self.xCoord, self.yCoord, self.altitude])

            self.progress = int(
                100 - ((current_dist / self.dist_to_dest) * 100))

            self.time = current_dist / self.velocity

            response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
            np_response_image = np.asarray(
                bytearray(response_image), dtype="uint8")
            decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
            _, self.encoded_img = cv2.imencode(DECODE_EXTENSION, decoded_frame)
            self.encoded_img = base64.b64encode(self.encoded_img)

            depth = depth_finder.get_depth_map(decoded_frame)
            normalizedImg = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX)

            ret, thresh = cv2.threshold(
                normalizedImg, 175, np.amax(normalizedImg), cv2.THRESH_BINARY)

            height, width, _ = decoded_frame.shape

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
                # print("TOO CLOSE TO OBJECT - STOPPING AND HOVERING")
                client.moveByVelocityAsync(0, 0, 0, 1).join()
                client.hoverAsync().join()
                # print("TAKING EVASIVE MANOUVER")
                self.evasive_manouver(crop_img)
                # print("done")
                client.moveToPositionAsync(
                    self.xCoord, self.yCoord, self.altitude, self.velocity)

            if (self.progress > 95):
                client.moveByVelocityAsync(0, 0, 0, 1).join()
                self.status = "Landing"
                self.arrived = True
                self.progress = 100
                landing_thread.start()

        landing_thread.join()

        home = False

        self.dist_to_dest = distance.euclidean(
            [self.xCoord, self.yCoord, self.altitude], [0, 0, self.altitude])

        self.status = "Initialising"

        client.moveToZAsync(self.altitude, 2).join()
        client.moveToPositionAsync(
            0, 0, self.altitude, self.velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)

        self.status = "Going Home"

        while not home:
            position = client.simGetVehiclePose().position

            current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                0, 0, self.altitude])

            self.progress = int(
                100 - ((current_dist / self.dist_to_dest) * 100))

            self.time = current_dist / self.velocity

            response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
            np_response_image = np.asarray(
                bytearray(response_image), dtype="uint8")
            decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
            _, self.encoded_img = cv2.imencode(DECODE_EXTENSION, decoded_frame)
            self.encoded_img = base64.b64encode(self.encoded_img)

            if (self.progress > 95):
                client.moveByVelocityAsync(0, 0, 0, 1).join()
                self.status = "Landing"
                self.progress = 100
                home = True

        client.moveToZAsync(-3, 1).join()
        client.landAsync().join()
        self.status = "Home"

        return


drone_controller = DroneController()

# model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23,
#                  activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

# model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', encoder_weights='imagenet', classes=23,
#                           activation=None, encoder_depth=5)

# model = torch.load('../models/Unet-Mobilenet.pt')

model = torch.load('../models/DeepLabV3Plus-Mobilenet.pt')

depth_finder = DepthFinder("mono+stereo_640x192")

landing_thread = threading.Thread(target=drone_controller.landing)

navigation_thread = threading.Thread(target=drone_controller.navigate)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def frame_generator():
    CAMERA_NAME = '0'
    IMAGE_TYPE = airsim.ImageType.Scene
    DECODE_EXTENSION = '.jpg'
    while (True):
        response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
        np_response_image = np.asarray(
            bytearray(response_image), dtype="uint8")
        decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
        ret, encoded_jpeg = cv2.imencode(DECODE_EXTENSION, decoded_frame)
        # frame = encoded_jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_jpeg) + b'\r\n')


@app.get('/')
async def root():
    return {"message": "Server Running"}


@app.post('/sendFlightParams')
async def set_flight_params(drone_controller_temp: DroneController):

    drone_controller.xCoord = drone_controller_temp.xCoord
    drone_controller.yCoord = drone_controller_temp.yCoord
    drone_controller.altitude = 0 - drone_controller_temp.altitude
    drone_controller.velocity = drone_controller_temp.velocity

    navigation_thread.start()

    status = {"status": "OK"}

    return status


# @app.get('/bottom_cam_image')
# async def get_bottom_image():
#     CAMERA_NAME = '0'
#     IMAGE_TYPE = airsim.ImageType.Scene
#     DECODE_EXTENSION = '.png'

#     response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
#     np_response_image = np.asarray(
#         bytearray(response_image), dtype="uint8")
#     decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
#     _, encoded_img = cv2.imencode(DECODE_EXTENSION, decoded_frame)
#     encoded_img = base64.b64encode(encoded_img)
#     return {"img": encoded_img}


@app.get('/get_trip_details')
async def get_status():

    status = drone_controller.get_status()

    return status
