# Copyright (c) 
import torch.utils.data
import torchvision

from .build import build as build_data

def build_dataset(image_set, args):
    return build_data(image_set, args)

