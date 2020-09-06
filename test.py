import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm 
import cv2 

from torch_dreams import  utils
from torch_dreams import dreamer
from torch_dreams.simple import vgg19_dreamer



mode = "vgg"

image_main = cv2.imread("torch_dreams/sample_images/camo.jpg")
image_sample = cv2.resize(image_main, (256,256))

plt.imshow(image_sample)
plt.show()


if mode == "vgg":
    model= models.vgg19(pretrained=True)
    layers = list(model.features.children())
    model.eval()

    preprocess = utils.preprocess_func ## for some reason this works
    deprocess = None
    layer = layers[13]


else:
    model = models.resnet18(pretrained=True)
    layers = list(model.children())
    model.eval()

    preprocess = utils.preprocess_func
    deprocess = None

    layer = layers[8]

dreamer = dreamer(model, preprocess, deprocess)


dreamed = dreamer.deep_dream(
                        image_np =image_sample, 
                        layer = layer, 
                        octave_scale = 1.3, 
                        num_octaves = 5, 
                        iterations = 7, 
                        lr = 0.003,
                        )

plt.imshow(dreamed)
plt.show()
cv2.imwrite("dream_1.jpg", dreamed)

dreamer.progressive_deep_dream(
    image_np = image_sample,
    save_name = "progressive_dream.mp4",
    layer = layer,
    octave_scale = 1.3,
    num_octaves = 1,
    iterations = 2,
    lower_lr = 0.0,
    upper_lr = 0.09,
    num_steps = 10,
    framerate = 5,
    size = (256,256)
)




"""
Simple dreamer
"""
simple_dreamer = vgg19_dreamer()

dreamed_image = simple_dreamer.dream(
    image_path = "torch_dreams/sample_images/camo.jpg",
    layer_index= 17,
    iterations= 7,
    size = (256,256),
    lr = 0.03, 
    num_octaves= 5,
    octave_scale= 1.3
)


plt.imshow(dreamed_image)
plt.show()
cv2.imwrite("dream_2_from_png.jpg", dreamed_image)

simple_dreamer.progressive_dream(
    image_path = "torch_dreams/sample_images/camo.jpg",
    save_name = "progressive_dream_simple.mp4",
    layer_index = 13,
    octave_scale = 1.3,
    num_octaves = 1,
    iterations = 2,
    lower_lr = 0.0,
    upper_lr = 0.09,
    num_steps = 10,
    framerate = 5,
    size = (256,256)
)



simple_dreamer.deep_dream_on_video(
    video_path = "sample_videos/tiger_mini.mp4",
    save_name = "dream.mp4",
    layer = simple_dreamer.layers[13],
    octave_scale= 1.3,
    num_octaves = 2,
    iterations= 2,
    lr = 0.09,
    size = None, 
    framerate= 30.0,
    skip_value =  1
)
