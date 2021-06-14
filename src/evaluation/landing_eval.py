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
from random import randint, choice
from scipy.spatial import distance
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
        landing_zone_paved_area = dict_percentages[(128, 64, 128)]
    except:
        landing_zone_paved_area = 0.0

    try:
        landing_zone_grass = dict_percentages[(0, 102, 0)]
    except:
        landing_zone_grass = 0.0

    landing_zone_percentage = landing_zone_paved_area + landing_zone_grass
    # print(landing_zone_percentage)
    return landing_zone_percentage


def reset_drone(client):
    client.moveByVelocityAsync(0, 0, 0, 1).join()
    client.hoverAsync().join()
    client.reset()
    time.sleep(2)
    client.enableApiControl(False)
    client.armDisarm(False)
    client.enableApiControl(True)
    client.armDisarm(True)


# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

bad_landings_unet = 0
bad_landings_deeplab = 0
bad_landings_nosyst = 0

safe_landing_surfaces = ["Driveway", "Road", "Landscape", "driveway"]

bad_landing_surfaces_unet = []
bad_landing_surfaces_deeplab = []
bad_landing_surfaces_nosyst = []

distance_deeplab = []
distance_unet = []
distance_nosyst = []

for i in range(100):

    r = choice([(-50, -15), (15, 50)])

    x = randint(*r)

    r = choice([(-60, -15), (15, 60)])

    y = randint(*r)

    for j in range(3):

        if j == 0:
            model = torch.load('models/DeepLabV3Plus-Mobilenetv2.pt')
        else:
            model = torch.load('models/UNET-Mobilenetv2.pt')

        destination_x = x
        destination_y = y

        client.moveToPositionAsync(0, 0, -30, 5).join()

        client.moveToPositionAsync(
            x, y, -30, 3).join()

        client.moveByVelocityAsync(0, 0, 0, 1).join()

        landed = False
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        search_count = 0

        while not landed:
            if j != 2:
                img = airsim.string_to_uint8_array(
                    client.simGetImage("bottom_center", airsim.ImageType.Scene))

                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                img = cv2.resize(img, (1056, 704))

                img2show = img.copy()

                pred_mask = predict_image(model, img)
                pred_mask = torch.argmax(
                    pred_mask.squeeze(), dim=0).detach().cpu().numpy()

                pred_mask = decode_segmap(pred_mask)

                height, width, _ = img.shape

                new_height = 100
                new_width = 100

                upper_left = (int((width - new_width) // 2),
                              int((height - new_height) // 2))
                bottom_right = (int((width + new_width) // 2),
                                int((height + new_height) // 2))

                img = pred_mask[upper_left[1]: bottom_right[1],
                                upper_left[0]: bottom_right[0]].copy()

                landing_zone_percent = percent_landing_zone(img)
                
                if landing_zone_percent > 95.0:
                    client.moveToPositionAsync(
                        destination_x, destination_y, -10, 5).join()
                    client.enableApiControl(False)
                    client.armDisarm(False)
                    time.sleep(3)
                    client.enableApiControl(True)
                    client.armDisarm(True)
                    object_name = client.simGetCollisionInfo().object_name
                    # print("Object Name: " + str(object_name))
                    landing_surface = object_name.split("_")[0]
                    if landing_surface not in safe_landing_surfaces:
                        if j == 0:
                            bad_landings_deeplab += 1
                            bad_landing_surfaces_deeplab.append(
                                landing_surface)
                        else:
                            bad_landings_unet += 1
                            bad_landing_surfaces_unet.append(landing_surface)

                    position = client.simGetVehiclePose().position
                    current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                        x, y, 0])

                    if j == 0:
                        distance_deeplab.append(current_dist)
                    else:
                        distance_unet.append(current_dist)

                    reset_drone(client)
                    break
                else:
                    search_count += 1
                    height, width, _ = pred_mask.shape

                    bottom_left = pred_mask[(height // 2):height,
                                            0:(width // 2)].copy()
                    bottom_right = pred_mask[(height // 2):height,
                                             (width // 2):width].copy()
                    top_left = pred_mask[0:(
                        height // 2), 0:(width // 2)].copy()
                    top_right = pred_mask[0:(height // 2),
                                          (width // 2):width].copy()

                    top_left_percent = percent_landing_zone(top_left)
                    top_right_percent = percent_landing_zone(top_right)
                    bottom_left_percent = percent_landing_zone(bottom_left)
                    bottom_right_percent = percent_landing_zone(bottom_right)

                    if top_left_percent > max(top_right_percent, bottom_left_percent, bottom_right_percent):
                        # print("Moving Top left")
                        destination_x += 2
                        destination_y += -2
                    elif top_right_percent > max(top_left_percent, bottom_left_percent, bottom_right_percent):
                        # print("Moving Top right")
                        destination_x += 2
                        destination_y += 2
                    elif bottom_left_percent > max(top_left_percent, top_right_percent, bottom_right_percent):
                        # print("Moving bottom left")
                        destination_x += -2
                        destination_y += -2
                    elif bottom_right_percent > max(top_left_percent, top_right_percent, bottom_left_percent):
                        # print("Moving bottom right")
                        destination_x += -2
                        destination_y += 2

                    if search_count >= 25:
                        client.moveToPositionAsync(
                            destination_x, destination_y, -10, 5).join()
                        client.enableApiControl(False)
                        client.armDisarm(False)
                        time.sleep(3)
                        client.enableApiControl(True)
                        client.armDisarm(True)
                        object_name = client.simGetCollisionInfo().object_name
                        # print("Object Name: " + str(object_name))
                        landing_surface = object_name.split("_")[0]
                        if landing_surface not in safe_landing_surfaces:
                            if j == 0:
                                bad_landings_deeplab += 1
                                bad_landing_surfaces_deeplab.append(
                                    landing_surface)
                            else:
                                bad_landings_unet += 1
                                bad_landing_surfaces_unet.append(
                                    landing_surface)

                        position = client.simGetVehiclePose().position
                        current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                            x, y, 0])

                        if j == 0:
                            distance_deeplab.append(current_dist)
                        else:
                            distance_unet.append(current_dist)
                        reset_drone(client)
                        break
                    client.moveToPositionAsync(
                        destination_x, destination_y, -30, 1).join()
                    client.moveByVelocityAsync(0, 0, 0, 1).join()
            else:
                client.moveToPositionAsync(
                    destination_x, destination_y, -10, 5).join()
                client.enableApiControl(False)
                client.armDisarm(False)
                time.sleep(3)
                client.enableApiControl(True)
                client.armDisarm(True)
                object_name = client.simGetCollisionInfo().object_name
                # print("Object Name: " + str(object_name))
                landing_surface = object_name.split("_")[0]
                if landing_surface not in safe_landing_surfaces:
                    bad_landings_nosyst += 1
                    bad_landing_surfaces_nosyst.append(landing_surface)

                position = client.simGetVehiclePose().position
                current_dist = distance.euclidean([position.x_val, position.y_val, position.z_val], [
                    x, y, 0])

                distance_nosyst.append(current_dist)
                reset_drone(client)
                break
    print("Loop No: " + str(i + 1))
    print("Bad Landings DeepLab: " + str(bad_landings_deeplab))
    print("Average Dist DeepLab: " +
          str(sum(distance_deeplab) / len(distance_deeplab)))
    print("Bad Landings UNET: " + str(bad_landings_unet))
    print("Average Dist UNET: " + str(sum(distance_unet) / len(distance_unet)))
    print("Bad Landings No system: " + str(bad_landings_nosyst))
    print("Average Dist No system: " +
          str(sum(distance_nosyst) / len(distance_nosyst)))

    if len(bad_landing_surfaces_deeplab) > 0:
        surface_freq = Counter(bad_landing_surfaces_deeplab)
        print("DEEPLAB: ")
        print(surface_freq)

    if len(bad_landing_surfaces_unet) > 0:
        surface_freq = Counter(bad_landing_surfaces_unet)
        print("UNET: ")
        print(surface_freq)

    if len(bad_landing_surfaces_nosyst) > 0:
        surface_freq = Counter(bad_landing_surfaces_nosyst)
        print("No System: ")
        print(surface_freq)


print("COMPLETE")
print("Bad Landings DeepLab: " + str(bad_landings_deeplab))
print("Average Dist DeepLab: " +
      str(sum(distance_deeplab) / len(distance_deeplab)))
print("Bad Landings UNET: " + str(bad_landings_unet))
print("Average Dist UNET: " + str(sum(distance_unet) / len(distance_unet)))
print("Bad Landings No system: " + str(bad_landings_nosyst))
print("Average Dist No system: " +
      str(sum(distance_nosyst) / len(distance_nosyst)))
surface_freq = Counter(bad_landing_surfaces_deeplab)
print("DEEPLAB: ")
print(surface_freq)
surface_freq = Counter(bad_landing_surfaces_unet)
print("UNET: ")
print(surface_freq)
surface_freq = Counter(bad_landing_surfaces_nosyst)
print("No System: ")
print(surface_freq)
