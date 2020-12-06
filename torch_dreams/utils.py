import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
from tqdm import tqdm 
import cv2 
import random

import torch.nn as nn

import numbers
import math
import torch.nn.functional as F

from .constants import IMAGENET_MEAN_1
from .constants import IMAGENET_STD_1
from .constants import IMAGENET_MEAN_1_GRAY
from .constants import IMAGENET_STD_1_GRAY

from .image_transforms import transform_to_tensor
from .image_transforms import rot_img

def load_image(img_path, target_shape=None, grayscale = False):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv2.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range

    if grayscale is True:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img

def pytorch_input_adapter(img, device):
    tensor = transform_to_tensor(img).to(device)
    return tensor

def get_random_rotation_angle(theta_max = 0.15):
    theta = (random.random() - 0.5) * theta_max
    # print(theta)
    return theta

def rotate_image_tensor(image_tensor, theta, device ):
    """
    when trying to use grayscale, it throws a NotImplementedError
    somehow, the single channel image is becoming 2D instead of 1D on 2nd iteration 

    refer issue (Fixed){
        https://github.com/Mayukhdeb/torch-dreams/issues/7
    }
    """
    image_rotated = rot_img(x = image_tensor.unsqueeze(0), theta = theta, device = device)
    return image_rotated.squeeze(0)

def pytorch_output_adapter(img):
    return np.moveaxis(img.to('cpu').detach().numpy()[0], 0, 2)


def preprocess_numpy_img(img, grayscale = False):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'

    if grayscale is False:

        img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1  # normalize image
    else:
        img = (img - IMAGENET_MEAN_1_GRAY) / IMAGENET_STD_1_GRAY  # normalize image
    return img

def post_process_numpy_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    if dump_img.shape[0] == 3:  # if channel-first format move to channel-last (CHW -> HWC)
        dump_img = np.moveaxis(dump_img, 0, 2)

    mean = IMAGENET_MEAN_1.reshape(1, 1, -1)
    std = IMAGENET_STD_1.reshape(1, 1, -1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = np.clip(dump_img, 0., 1.)

    return dump_img

    
def find_random_roll_values_for_tensor(image_tensor):

    """
    image_tensor.size() should be (C, H, W)
    """

    max_roll_x = image_tensor.size()[-1]
    max_roll_y = image_tensor.size()[-2]

    roll_x = random.randint(-max_roll_x, max_roll_x)
    roll_y = random.randint(-max_roll_y, max_roll_y)

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

class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing seperately for each channel (depthwise convolution)
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size, sigma, device, grayscale=False):
        super().__init__()
        self. device = device
        self.grayscale = grayscale
        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers

        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size / 2)  # Used to pad the channels so that after applying the kernel we have same size

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        for sigma in sigmas:
            kernel = 1
            for size_1d, std_1d, mgrid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        prepared_kernels = []
        for kernel in kernels:
            # Make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)

            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            if self.grayscale is False:
                kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
            else:
                kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

            kernel = kernel.to(self.device)
            prepared_kernels.append(kernel)

        self.register_buffer('weight1', prepared_kernels[0])
        self.register_buffer('weight2', prepared_kernels[1])
        self.register_buffer('weight3', prepared_kernels[2])
        self.conv = F.conv2d

    def forward(self, input):
        """
        Apply gaussian filter to input.
        """
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        if self.grayscale is False:
            grad1 = self.conv(input, weight=self.weight1, groups=3)
            grad2 = self.conv(input, weight=self.weight2, groups=3)
            grad3 = self.conv(input, weight=self.weight3, groups=3)
            return grad1 + grad2 + grad3
        else:
            grad1 = self.conv(input, weight=self.weight1, groups=1)
            grad2 = self.conv(input, weight=self.weight2, groups=1)
            grad3 = self.conv(input, weight=self.weight3, groups=1)
            return grad1 + grad2 + grad3

    