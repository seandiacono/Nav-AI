import airsim


class DroneController():

    def __init__(self, x, y, altitude, velocity):
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
            self.client.takeoffAsync().join()
            self.flying = True

        # rise to altitude
        print(self.altitude)
        self.client.enableApiControl(True)
        self.client.moveToPositionAsync(0, 0, self.altitude, 2).join()

        self.client.enableApiControl(True)
        self.client.moveToPositionAsync(
            self.x, self.y, self.altitude, self.velocity, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20).join()


drone_controller = DroneController(30, 30, -30, 2)

print("navigation")
drone_controller.navigate()
