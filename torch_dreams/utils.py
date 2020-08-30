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


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


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
    image_tensor.size() should be (width, height, color_channels)
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


def get_gradients(net_in, net, layer, out_channels = None):     
    net_in = net_in.unsqueeze(0)
    net_in.requires_grad = True
    net.zero_grad()
    hook = Hook(layer)
    net_out = net(net_in)
    if out_channels == None:
        loss = hook.output[0].norm()
    else:
        loss = hook.output[0][out_channels].norm()
    loss.backward()
    return net_in.grad.data.squeeze()