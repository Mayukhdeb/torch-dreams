import torch
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

from torch_dreams.utils import tensor_to_image


image_main = cv2.imread("sample.jpg")
image_sample = cv2.resize(image_main, (512,512))

# plt.imshow(image_sample)
# plt.show()

model= models.vgg19(pretrained=True)
layers = list(model.features.children())
model.eval()

preprocess = utils.preprocess_func ## for some reason this works
layer = layers[24]

dreamy_boi = dreamer(model = model)

image_tensor = preprocess(image_sample).unsqueeze(0)

out_tensor = dreamy_boi.deep_dream(
    image_tensor = image_tensor,
    layer = layer,
    octave_scale = 1.3,
    num_octaves = 7,
    iterations = 5,
    lr = 0.05
)

img_out_np = tensor_to_image(out_tensor)

plt.imshow(cv2.cvtColor(img_out_np, cv2.COLOR_BGR2RGB))
plt.show()