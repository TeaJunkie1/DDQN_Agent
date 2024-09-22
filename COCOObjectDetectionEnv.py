import os
import random
import numpy as np
import cv2
from pycocotools.coco import COCO
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class COCOObjectDetectionEnv:
    def __init__(self, coco_annotation_file, coco_image_dir, image_size=(128, 128), box_size=(64, 64)):
        self.coco = COCO(coco_annotation_file)
        self.coco_image_dir = coco_image_dir
        self.image_size = image_size
        self.box_size = box_size
        self.image_ids = list(self.coco.imgs.keys())
        self.valid_image_ids = [img_id for img_id in self.image_ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        self.current_image = None
        self.current_label = None
        self.current_box = None
        self.initial_box = [0, 0, box_size[0], box_size[1]]

    def reset(self):
        idx = np.random.randint(len(self.valid_image_ids))
        img_id = self.valid_image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        img_path = os.path.join(self.coco_image_dir, img_info['file_name'])
        image = np.array(Image.open(img_path))

        bbox = anns[0]['bbox']
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]

        self.current_image = cv2.resize(image, self.image_size)
        self.current_label = bbox
        self.current_box = self.initial_box.copy()

        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        reward = self._compute_reward()
        done = self._is_done()
        return self._get_observation(), reward, done, {}

    def _take_action(self, action):
        step_size = 10
        if action == 0:  # move left
            self.current_box[0] = max(0, self.current_box[0] - step_size)
        elif action == 1:  # move right
            self.current_box[0] = min(self.image_size[0] - self.box_size[0], self.current_box[0] + step_size)
        elif action == 2:  # move up
            self.current_box[1] = max(0, self.current_box[1] - step_size)
        elif action == 3:  # move down
            self.current_box[1] = min(self.image_size[1] - self.box_size[1], self.current_box[1] + step_size)
        elif action == 4:  # expand
            self.current_box[2] = min(self.image_size[0], self.current_box[2] + step_size)
            self.current_box[3] = min(self.image_size[1], self.current_box[3] + step_size)
        elif action == 5:  # shrink
            self.current_box[2] = max(self.box_size[0], self.current_box[2] - step_size)
            self.current_box[3] = max(self.box_size[1], self.current_box[3] - step_size)

    def _compute_reward(self):
        box_pred = self.current_box
        box_true = self.current_label

        xA = max(box_pred[0], box_true[0])
        yA = max(box_pred[1], box_true[1])
        xB = min(box_pred[2], box_true[2])
        yB = min(box_pred[3], box_true[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        boxBArea = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _is_done(self):
        return self._compute_reward() > 0.7

    def _get_observation(self):
        return self.current_image

# Initialize the environment
