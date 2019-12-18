import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    """Flips image horizontally"""
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def vertical_flip(images, targets):
    """Flips image vertically"""
    images = torch.flip(images, [-2])
    targets[:, 1] = 1 - targets[:, 1]
    return images, targets


def gaussian_noise(images, targets):
    """Adds Gaussian noise"""
    noise = torch.rand_like(images)
    noisy_images = images + noise
    torch.clamp(noisy_images, 0, 1)
    return noisy_images, targets


def multiply(images, targets):
    """Multiplies by random coefficient in range [0, 2]"""
    coeff = np.random.randint(0, 2)
    multiplied_images = images * coeff
    return multiplied_images, targets


def salt_and_pepper(images, targets):
    """Adds salt and pepper noise"""
    prob = 0.1
    rnd = np.random.rand(images.shape)
    images[rnd < prob/2] = 0
    images[rnd > 1 - prob/2] = 1
    return images, targets
