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

    model: Optional[smp.Unet] = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23,
                                         activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

    model = torch.load('../models/Unet-Mobilenet.pt')

    def predict_image(self, model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        model.eval()
        t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        image = t(image)
        model.to(device)
        image = image.to(device)
        with torch.no_grad():
            image = image.unsqueeze(0)
            output = model(image)
            # masked = torch.argmax(output, dim=1)
            # masked = masked.cpu().squeeze(0)
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
        img = airsim.string_to_uint8_array(
            client.simGetImage("bottom_center", airsim.ImageType.Scene))

        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, (1056, 704))

        pred_mask = self.predict_image(self.model, img)
        pred_mask = torch.argmax(
            pred_mask.squeeze(), dim=0).detach().cpu().numpy()

        pred_mask = self.decode_segmap(pred_mask)

        # cv2.imshow("mask", pred_mask)
        # cv2.waitKey(1)

        height, width, _ = img.shape

        new_height = 64
        new_width = 64

        upper_left = (int((width - new_width) // 2),
                      int((height - new_height) // 2))
        bottom_right = (int((width + new_width) // 2),
                        int((height + new_height) // 2))

        img = pred_mask[upper_left[1]: bottom_right[1],
                        upper_left[0]: bottom_right[0]]

        img = np.asarray([img])

        mode = self.bincount_app(img)

        if mode == (128, 64, 128):
            client.landAsync()
            landed = True
        else:
            d = None
            # x -= 1
            # client.moveToPositionAsync(x, y, -30, 0.5).join()

    def get_status(self):
        position = client.simGetVehiclePose().position

        current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
            self.xCoord, self.yCoord, self.altitude])

        prog_percentage = int(100-((current_dist/self.dist_to_dest) * 100))

        status = "Flying"

        if (prog_percentage > 95):
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            status = "Arrived"
            prog_percentage = 100

        time = current_dist / self.velocity

        CAMERA_NAME = '0'
        IMAGE_TYPE = airsim.ImageType.Scene
        DECODE_EXTENSION = '.jpeg'

        response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
        np_response_image = np.asarray(
            bytearray(response_image), dtype="uint8")
        decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
        _, encoded_img = cv2.imencode(DECODE_EXTENSION, decoded_frame)
        encoded_img = base64.b64encode(encoded_img)

        return {"progress": prog_percentage, "time_left": int(time), "status": status, "image": encoded_img}

    def navigate(self):

        client.armDisarm(True)
        # take off
        print("taking off")
        client.takeoffAsync()

        # rise to altitude
        print(self.altitude)
        client.enableApiControl(True)
        client.moveToPositionAsync(0, 0, self.altitude, 2).join()

        self.dist_to_dest = distance.euclidean(
            [0, 0, self.altitude], [self.xCoord, self.yCoord, self.altitude])

        client.moveToPositionAsync(
            self.xCoord, self.yCoord, self.altitude, self.velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)


drone_controller = DroneController()

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
    drone_controller.altitude = drone_controller_temp.altitude
    drone_controller.velocity = drone_controller_temp.velocity

    drone_controller.navigate()

    status = {"status": "OK"}

    return status


@app.get('/videoStream')
async def get_video_stream():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get('/bottom_cam_image')
async def get_bottom_image():
    CAMERA_NAME = '0'
    IMAGE_TYPE = airsim.ImageType.Scene
    DECODE_EXTENSION = '.png'

    response_image = client.simGetImage(CAMERA_NAME, IMAGE_TYPE)
    np_response_image = np.asarray(
        bytearray(response_image), dtype="uint8")
    decoded_frame = cv2.imdecode(np_response_image, cv2.IMREAD_COLOR)
    _, encoded_img = cv2.imencode(DECODE_EXTENSION, decoded_frame)
    encoded_img = base64.b64encode(encoded_img)
    return {"img": encoded_img}


@app.get('/get_trip_details')
async def get_status():

    status = drone_controller.get_status()

    return status
