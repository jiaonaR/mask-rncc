# from mrcnn.utils import compute_ap
# from mrcnn.model import load_image_gt
# from mrcnn.model import mold_image
# from mrcnn.model import MaskRCNN
# from mrcnn.config import Config
# from mrcnn.utils import Dataset
# from mrcnn.visualize import display_instances
# from mrcnn.utils import extract_bboxes

# # Create model object in inference mode.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import pycocotools
import tensorflow as tf

from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

class CarConfig(Config):
  # Give the configuration a recognizable name
  NAME = "car_cfg"
  # Number of classes (background + vehicle)
  NUM_CLASSES = 1 + 1
  # Number of training steps per epoch
  STEPS_PER_EPOCH = 50
  IMAGE_MAX_DIM = 640

# prepare config
config = CarConfig()
# config.display()
# model = MaskRCNN(mode='inference', model_dir='/content/', config=config)

# Load weights trained on MS-COCO
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights('mask_rcnn_coco.h5', by_name=True)
# exit(0)
class PredictionConfig(Config):
    NAME = "car_cfg"
    NUM_CLASSES = 1+3
    IMAGE_SHAPE = [512,512,3]
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def evaluate_model(dataset, model, cfg):
    APs = list()
    for i,image_id in enumerate(dataset.image_ids):
    # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
        if i > 10: # EVALUATE ONLY 10 IMAGES, DUE TO TIME
            break
    mAP = np.mean(APs)
    return mAP

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
carind = class_names.index('car')
cfg = PredictionConfig()
# cfg.IMAGE_SHAPE = [1024,512,3]
cfg.display()
# print(cfg.IMAGE_SHAPE)
# exit(0)
# define the model
model = MaskRCNN(mode='inference', model_dir='/content/', config=cfg)
# load model weights
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]) #path to model in colab
# model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]) #path to model in colab
# exit(0)
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
file_names = './images/221.JPG'
image = skimage.io.imread(file_names)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
#
# r = results[1]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# r = results[2]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# r = results[3]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])




