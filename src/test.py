import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

x = 20
y = 8
z = -10

# Async methods returns Future. Call join() to wait for task to complete.
print("TAKING OFF")
client.moveToPositionAsync(0, 0, z, 2).join()
print("TAKEOFF COMPLETE")

client.moveToPositionAsync(x, y, z, 2, yaw_mode=airsim.YawMode(is_rate=False,
                                                               yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20).join()

client.moveByVelocityBodyFrameAsync(0, -2, 0, 2).join()

# client.moveToPositionAsync(x, y, z, 2)

# client.moveByVelocityAsync(0, 0, 0, 1)

# result = client.getMultirotorState()

# print(result)
