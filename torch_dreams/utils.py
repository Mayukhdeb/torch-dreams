import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2 

def preprocess_func_vgg(image_np):

    """
    specific for VGG19
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                     ])
    return preprocess(image_np)

def preprocess_func(image_np):

    """
    simple version
    """

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    preprocess = transforms.Compose([
                                     transforms.ToTensor()
                                    #  transforms.Normalize(mean, std)
                                     ])
    return preprocess(image_np)


def deprocess_func_vgg(image_tensor):

    """
    specific for VGG19
    """

    denorm = transforms.Compose([ 
                                 transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                                ])
    return denorm(image_tensor)

def deprocess_func(image_tensor):


    denorm = transforms.Compose([ 
                                 transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                 transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ], std = [ 1., 1., 1. ]),                                                     
                                ])
    return denorm(image_tensor)

def find_random_roll_values_for_tensor(image_tensor):

    """
    image_tensor.size() should be (C, H, W)
    """

    roll_x = image_tensor.size()[-1]
    roll_y = image_tensor.size()[-2]

    return roll_x, roll_y


def roll_torch_tensor(image_tensor, roll_x, roll_y):

    """
    rolls a torch tensor on both x and y axis 
    """

    rolled_tensor = torch.roll(torch.roll(image_tensor, shifts = roll_x, dims = -1), shifts = roll_y, dims = -2)

    return rolled_tensor

def video_to_np_arrays(video_path, skip_value = 1, size = None):

    vidObj = cv2.VideoCapture(video_path)   
    success = 1
    images = []
    count = 0

    while success: 
        count +=1 
        success, image = vidObj.read() 

        if count % skip_value != 0:
            continue
        else:
            try:
                if size is not None:
                    image = cv2.resize(image, size)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                    
            except:
                pass
    
    return np.array(images)

def write_video_from_image_list(save_name, all_images_np, framerate, size):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(save_name ,fourcc, framerate, size)

    for i in range(all_images_np.shape[0]):
        
        frame = all_images_np[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out.write(frame)
    out.release()
    