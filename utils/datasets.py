import glob
import random
import os
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class PandasDataset(Dataset):
    def __init__(self, data_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        all_masks = PandasDataset.get_all_mask_names(data_path)
        rows = []
        columns = ["exp_name", "image_path", "mask_name", "mask_path", "label_id", "gt_x1", "gt_y1", "gt_x2", "gt_y2", "width", "height"]

        for image_masks in all_masks:
            for mask in image_masks.get("exp_masks"):

                mask_path = os.path.join(data_path, image_masks.get("exp_name"), 'masks', mask["name"])
                image = cv2.imread(mask_path)
                gt_x1, gt_y1, gt_x2, gt_y2 = PandasDataset.get_bbox(image)
                row = (
                    image_masks.get("exp_name"), 
                    image_masks.get("image_path"), 
                    mask.get("name"), 
                    mask_path, 
                    mask.get("label_id"), 
                    gt_x1, 
                    gt_y1, 
                    gt_x2, 
                    gt_y2, 
                    image.shape[0], 
                    image.shape[1]
                    )
                rows.append(row)

        self.data = pd.DataFrame.from_records(rows, columns=columns)
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
    
    def __getitem__(self, index):

        img_path = self.data.iloc[index % len(self.data)]["img_path"]
        exp_name = self.data.iloc[index % len(self.data)]["exp_name"]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        targets = None
        
        boxes = np.array(self.data.loc[self.data["exp_name"] == exp_name].filter(items=["label_id","gt_x1","gt_x2","gt_y1","gt_y2"]), dtype='f')
        x1, y1, x2, y2 = boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4]
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] = boxes[:, 3] * w_factor / padded_w
        boxes[:, 4] = boxes[:, 4] * h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = torch.from_numpy(boxes)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def __len__(self):
        return len(self.data)

    def get_bbox(image):
        t = np.where(image == image.max())
        maxs = np.array(list(zip(t[0],t[1])))
        gt_x1, gt_y1 = np.min(maxs[:,1]), np.min(maxs[:,0])
        gt_x2, gt_y2 = np.max(maxs[:,1]), np.max(maxs[:,0])
        return gt_x1, gt_y1, gt_x2, gt_y2

    def get_all_mask_names(data_path):
        return [
            {"exp_name":dir, 
             "img_path":os.path.join(data_path, dir, 'images', dir + '.png'),
             "exp_masks":[{"name":mask_name,"label_id":0} for mask_name in os.listdir(os.path.join(data_path, dir, 'masks'))]
            } for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]
    
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            if np.random.random() < 0.5:
                img, targets = vertical_flip(img, targets)
            if np.random.random() < 0.1:
                img, targets = gaussian_noise(img, targets)
            if np.random.random() < 0.1:
                img, targets = multiply(img, targets)
            if np.random.random() < 0.1:
                img, targets = salt_and_pepper(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
