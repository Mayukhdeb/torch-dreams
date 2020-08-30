import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
]import os
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2 

from utils import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dream(image_np, net, layer, iterations, lr, preprocess_func, deprocess_func = None,  out_channels = None):

    image_tensor = preprocess_func(image_np).to(device)

    for i in tqdm(range(iterations)):

        roll_x, roll_y = find_random_roll_values_for_tensor(image_tensor)
        image_tensor_rolled = roll_torch_tensor(image_tensor, roll_x, roll_y) 
        gradients_tensor = get_gradients(image_tensor_rolled, net, layer, out_channels).detach()
        gradients_tensor = roll_torch_tensor(gradients_tensor, -roll_x, -roll_y)  
        image_tensor.data = image_tensor.data + lr * gradients_tensor.data ## can confirm this is still on the GPU if you have one

    img_out = image_tensor.detach().cpu()

    if deprocess_func is not None:
        img_out = deprocess_func(img_out)

    img_out_np = img_out.numpy()

    img_out_np = img_out_np.transpose(1,2,0)
    
    return img_out_np


def deep_dream(image_np, model, layer, octave_scale, num_octaves, iterations, lr, preprocess_func , deprocess_func = None):
    original_size = image_np.shape[:2]

    for n in range(-num_octaves, 1):
        
        octave_size = tuple( np.array(original_size) * octave_scale**n)
        new_size = (int(octave_size[1]), int(octave_size[0]))
        image_np = cv2.resize(image_np, new_size)
        image_np = dream(image_np, model, layer, iterations = iterations, lr = lr, out_channels = None, preprocess_func = preprocess_func, deprocess_func = deprocess_func)
        image_np_normalised = cv2.normalize(image_np, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)

    return image_np_normalised