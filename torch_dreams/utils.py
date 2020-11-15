import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tqdm
from torchvision import transforms
from tqdm import tqdm 
import cv2 

import random

from .constants import IMAGENET_MEAN_1
from .constants import IMAGENET_STD_1

transform_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])


def load_image(img_path, target_shape=None):
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
    return img

def pytorch_input_adapter(img, device):
    tensor = transforms.ToTensor()(img).to(device)
    return tensor


def pytorch_output_adapter(img):
    return np.moveaxis(img.to('cpu').detach().numpy()[0], 0, 2)


def preprocess_numpy_img(img):
    assert isinstance(img, np.ndarray), f'Expected numpy image got {type(img)}'

    img = (img - IMAGENET_MEAN_1) / IMAGENET_STD_1  # normalize image
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
    