from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
# from drone_controller import DroneController
import airsim

import asyncio

loop = asyncio.get_event_loop()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class FlightParams():

    def init_flight_params(self, x, y, altitude, velocity):
        self.x = x
        self.y = y
        self.altitude = altitude
        self.velocity = velocity

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.flying = False

    def navigate(self):

        self.client.armDisarm(True)
        # take off
        if not self.flying:
            print("taking off")
            self.client.takeoffAsync()
            self.flying = True

        # rise to altitude
        print(self.altitude)
        self.client.enableApiControl(True)
        result = self.client.moveToPositionAsync(0, 0, self.altitude, 2).join()
        response = result.result()

        self.client.moveToPositionAsync(
            self.x, self.y, self.altitude, self.velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)


flight_params = FlightParams()


@app.route('/')
@cross_origin()
def index():
    return "Server Running"


@app.route('/sendFlightParams', methods=['POST'])
@cross_origin()
def set_flight_params():
    data = request.get_json()

    x = data['xCoord']
    y = data['yCoord']
    altitude = data['altitude']
    velocity = data['velocity']

    # flight_params.init_flight_params(x, y, altitude, velocity)

    print("Creating Drone Controller")
    # drone_controller = DroneController(x, y, altitude, velocity)

    print("navigation")
    # flight_params.navigate()

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)

    client.takeoffAsync().join()
    print("Takeoff Complete")
    client.moveToPositionAsync(0, 0, altitude, 2).join()
    client.landAsync().join()

    status = {"status": "OK"}

    return status


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
