import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from torchvision import transforms


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1.,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
# creating a function to predict, plot the image with bounding boxes and masks aswell as printing total count of vehicles
def predict(img_path, cuda=True):
    vehicles_indices = [3]
    # setting the model to evaluation mode for inference
    model.eval()

    # loading PIL Image from path and creating a torch tensor from it
    img_pil = PIL.Image.open(img_path)
    img_tensor = transforms.functional.to_tensor(img_pil)

    # if GPU is wanted (from argument)
    if cuda:
        img_tensor = img_tensor.cuda()
        model.cuda()
    else:
        img_tensor = img_tensor.cpu()
        model.cpu()

    '''
    Making the prediction. The model returns a list with the results for the image. Inside the list there's a dictionary with
    "boxes", "labels", "scores" and "masks".
    The "boxes" are tensors with (x0, y0, x1, y1).
    '''
    predictions = model([img_tensor])

    # saving the image in cv2 to place bounding boxes later
    img_cv = cv2.imread(img_path)

    # creating a figure to plot later
    plt.figure(figsize=(15, 20))

    # setting vehicle count and masks placeholders
    n_vehicles = 0
    masks = []

    for i in range(predictions[0]["boxes"].shape[0]):
        # set the threshold for the prediction as you like. Here is 0.5
        if predictions[0]["scores"][i] > 0.5:
            label_id = predictions[0]["labels"][i].item()

            # check if the class predicted is a vehicle. If yes, get the bboxes and masks
            if label_id in vehicles_indices:
                bbox = predictions[0]["boxes"][i].detach().cpu().numpy()

                # draw a rectangle from the bbox on the cv2 image
                cv2.rectangle(img_cv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                mask = predictions[0]["masks"][i].detach().cpu().numpy().squeeze()

                # the model returns a mask with values from 0 to 1. Numpy masks ignoring every pixel that's lower than 0.5.
                _, mask = cv2.threshold(mask, 0.5, 1, 0)
                # mask = np.ma.masked_array(mask, mask <= 0.5)
                mask3d = np.array(mask)
                # mask3d = np.expand_dims(np.array(mask),axis=2).repeat(3,axis=2) *255.
                # cv2.addWeighted(img_cv, 0.5, mask3d, 0.5, 0)
                # cv2.addWeighted(img1, 0.5, cv2.resize(img2, (600, 300)), 0.5, 0)
                mm = apply_mask(img_cv,mask,(1,1,0))
                masks.append(mask)
                n_vehicles += 1
                plt.imshow(cv2.cvtColor(mm, cv2.COLOR_BGR2RGB))
                # plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                plt.show()
                plt.imshow(mask, alpha=.5, cmap="jet")
                plt.show()

    # plot the cv2 image
    # plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # plt.show()

    # plot all the masks found in the prediction
    # for mask in masks:
    #     plt.imshow(mask, alpha=.5, cmap="jet")
    #     plt.show()

    # print the total number of vehicles found
    print(f"Vehicle Count: {n_vehicles}")

if __name__ == '__main__':
    # loading pre-trained model. Using Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # From PyTorch documentation for the Mask R-CNN classes
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'N/A', 'stop',
        'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball',
        'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis',
        'racket', 'bottle', 'N/A', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'N/A', 'dining', 'table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell',
        'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush'
    ]

    # class indices corresponding to types of vehicles and bikes

    # vehicles_indices = [2, 3, 4, 6, 8]
    predict('./images/221.JPG', cuda=False)  # ADD INSIDE THE QUOTES YOUR IMAGE PATH