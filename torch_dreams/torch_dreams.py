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

from utils import *


image_main = cv2.imread("sample_images/cloudy-mountains.jpg")
image_sample = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
image_sample = cv2.resize(image_sample, (1024,1024))

plt.imshow(image_sample)
plt.show()

model = models.resnet18(pretrained=True)
layers = list(model.children())
model.eval()


def dream(image_np, net, layer, iterations, lr, preprocess_func, deprocess_func = None,  out_channels = None):

    image_tensor = preprocess_func(image_np)   ## removes .cuda()


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

    # img_out_np = (img_out_np -  img_out_np.min())/ (img_out_np.max() - img_out_np.min())
    
    return img_out_np


def deep_dream(image_np, model, layer, octave_scale, num_octaves, iterations, lr, preprocess_func , deprocess_func = None):
    original_size = image_np.shape[:2]

    for n in range(-num_octaves, 1):
        octave_size = tuple( np.array(original_size) * octave_scale**n)
        new_size = (int(octave_size[1]), int(octave_size[0]))

        image_np = cv2.resize(image_np, new_size)

        image_np = dream(image_np, model, layer, iterations = iterations, lr = lr, out_channels = None, preprocess_func = preprocess_func, deprocess_func = deprocess_func)

    return image_np

layer = layers[8]

dreamed = deep_dream(
                    image_sample, 
                    model,
                    layer = layer, 
                    octave_scale = 1.5, 
                    num_octaves = 7, 
                    iterations = 5, 
                    lr = 0.09,
                    preprocess_func = preprocess_func,
                    deprocess_func = None
                    )

plt.imshow(dreamed)
plt.show()

norm_dream = cv2.normalize(dreamed, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)


cv2.imwrite('dream.jpg', norm_dream.astype(np.uint8))