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


model = smp.Unet('mobilenet_v2', encoder_weights='imagenet', classes=23,
                 activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])

model = torch.load('models/Unet-Mobilenet.pt')

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
# client.moveToPositionAsync(0, 0, -30, 5)
print("FLYING TO DESTINATION")

x = -55
y = 23

client.moveToPositionAsync(x, y, -30, 2).join()

landed = False
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


while not landed:
    img = airsim.string_to_uint8_array(
        client.simGetImage("bottom_center", airsim.ImageType.Scene))

    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img, (1056, 704))

    pred_mask = predict_image(model, img)
    pred_mask = torch.argmax(pred_mask.squeeze(), dim=0).detach().cpu().numpy()

    pred_mask = decode_segmap(pred_mask)

    cv2.imshow("mask", pred_mask)
    cv2.waitKey(1)

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

    mode = bincount_app(img)

    print(mode)
    if mode == (128, 64, 128):
        print("LANDING")
        client.moveToPositionAsync(x, y, 1, 2).join()
        landed = True
    else:
        print("NOT SAFE TO LAND...MOVING TO NEW POSITION")
        x -= 1
        print(x)
        client.moveToPositionAsync(x, y, -30, 0.5).join()
        print("CHECKING NEW SPOT")
    # masked_img = cv2.addWeighted(img, 0.75, pred_mask, 0.25, 0)

    # cv2.imshow("img", masked_img)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    # ax1.imshow(img)
    # ax1.set_title('Picture')

    # ax2.imshow(pred_mask)
    # ax2.set_title('Predicted Mask')
    # ax2.set_axis_off()

    # ax3.imshow(img)
    # ax3.imshow(pred_mask, alpha=0.6)
    # ax3.set_title('Picture with Mask Appplied')
    # ax3.set_axis_off()

    # plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        client.cancelLastTask()
        client.moveByVelocityAsync(0, 0, 0, 1)
        client.hoverAsync()
        break


client.armDisarm(False)
client.enableApiControl(False)
