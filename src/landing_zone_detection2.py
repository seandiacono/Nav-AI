import numpy as np
import matplotlib.pyplot as plt
import airsim
import torch
from torchvision import transforms as T
import cv2
import time
import os
import segmentation_models_pytorch as smp
from PIL import Image
from scipy import stats
from collections import Counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
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


def decode_segmap(image, nc=23):

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


def bincount_app(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def percent_landing_zone(img):
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


model = torch.load('models/DeepLabV3Plus-MobileNet.pt')

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
print("TAKING OFF")
client.takeoffAsync()
print("TAKEOFF COMPLETE")
print("FLYING TO 40 ALTITUDE")
client.moveToPositionAsync(0, 0, -30, 5).join()
# orientation = client.simGetVehiclePose().orientation

# print(orientation)
print("FLYING TO DESTINATION")

x = 28
y = -18

destination_x = x
destination_y = y

client.moveToPositionAsync(
    x, y, -25, 2, yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0), drivetrain=airsim.DrivetrainType.ForwardOnly, lookahead=20)
time.sleep(3)


client.moveToPositionAsync(
    x, y, -25, 2).join()

client.moveByVelocityAsync(0, 0, 0, 1).join()

client.moveByRollPitchYawThrottleAsync(0.0, 0.0, 0.0, 0.5, 1).join()

landed = False
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


while not landed:
    img = airsim.string_to_uint8_array(
        client.simGetImage("bottom_center", airsim.ImageType.Scene))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (1056, 704))

    img2show = img.copy()

    pred_mask = predict_image(model, img)
    pred_mask = torch.argmax(pred_mask.squeeze(), dim=0).detach().cpu().numpy()

    pred_mask = decode_segmap(pred_mask)

    cv2.imshow("mask", pred_mask)
    cv2.waitKey(1)

    height, width, _ = img.shape

    new_height = 100
    new_width = 100

    upper_left = (int((width - new_width) // 2),
                  int((height - new_height) // 2))
    bottom_right = (int((width + new_width) // 2),
                    int((height + new_height) // 2))

    img = pred_mask[upper_left[1]: bottom_right[1],
                    upper_left[0]: bottom_right[0]].copy()

    cv2.imshow("cropped", img)
    cv2.waitKey(1)

    landing_zone_percent = percent_landing_zone(img)

    print(landing_zone_percent)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    # ax1.imshow(img2show)
    # ax1.set_title('Picture')

    # ax2.imshow(pred_mask)
    # ax2.set_title('Predicted Mask')
    # ax2.set_axis_off()

    # ax3.imshow(img)
    # ax3.imshow(pred_mask, alpha=0.6)
    # ax3.set_title('Picture with Mask Appplied')
    # ax3.set_axis_off()

    # plt.show()

    if landing_zone_percent > 95.0:
        print("LANDING")
        client.moveToPositionAsync(destination_x, destination_y, 1, 2).join()
        landed = True
    else:
        print("NOT SAFE TO LAND...MOVING TO NEW POSITION")
        height, width, _ = pred_mask.shape

        bottom_left = pred_mask[(height // 2):height, 0:(width // 2)].copy()
        bottom_right = pred_mask[(height // 2):height,
                                 (width // 2):width].copy()
        top_left = pred_mask[0:(height // 2), 0:(width // 2)].copy()
        top_right = pred_mask[0:(height // 2), (width // 2):width].copy()

        cv2.imshow("tl", top_left)
        cv2.imshow("tr", top_right)
        cv2.imshow("bl", bottom_left)
        cv2.imshow("br", bottom_right)
        cv2.waitKey(1)
        top_left_percent = percent_landing_zone(top_left)
        top_right_percent = percent_landing_zone(top_right)
        bottom_left_percent = percent_landing_zone(bottom_left)
        bottom_right_percent = percent_landing_zone(bottom_right)

        if top_left_percent > max(top_right_percent, bottom_left_percent, bottom_right_percent):
            print("Moving Top left")
            destination_x += 2
            destination_y += -2
        elif top_right_percent > max(top_left_percent, bottom_left_percent, bottom_right_percent):
            print("Moving Top right")
            destination_x += 2
            destination_y += 2
        elif bottom_left_percent > max(top_left_percent, top_right_percent, bottom_right_percent):
            print("Moving bottom left")
            destination_x += -2
            destination_y += -2
        elif bottom_right_percent > max(top_left_percent, top_right_percent, bottom_left_percent):
            print("Moving bottom right")
            destination_x += -2
            destination_y += 2

        client.moveToPositionAsync(
            destination_x, destination_y, -30, 1).join()
        client.moveByVelocityAsync(0, 0, 0, 1).join()
        print("CHECKING NEW SPOT")
    # masked_img = cv2.addWeighted(img, 0.75, pred_mask, 0.25, 0)

    # cv2.imshow("img", masked_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.cancelLastTask()
        client.moveByVelocityAsync(0, 0, 0, 1)
        client.hoverAsync()
        break
