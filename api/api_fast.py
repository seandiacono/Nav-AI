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
    dist_to_dest: Optional[float] = 0

    def get_status(self):
        position = client.simGetVehiclePose().position

        current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
            self.xCoord, self.yCoord, self.altitude])

        prog_percentage = int(100-((current_dist/self.dist_to_dest) * 100))

        status = "Flying"

        if (prog_percentage > 95):
            client.moveByVelocityAsync(0, 0, 0, 1).join()
            status = "Arrived"

        time = current_dist / self.velocity

        return {"progress": prog_percentage, "time_left": int(time), "status": status}

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
