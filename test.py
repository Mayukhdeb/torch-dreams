import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch_dreams.dreamer import dreamer
import torchvision.models as models

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

layers_to_use = [model.Mixed_6c.branch1x1]

def my_custom_func(layer_outputs):
    
    loss = layer_outputs[0].mean()
    return loss

config = {
    "image_path": "images/sample_small.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.2,
    "num_octaves": 10,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": my_custom_func,
    "max_rotation": 0.5,
    "grayscale": False,
    "gradient_smoothing_coeff": None,
    "gradient_smoothing_kernel_size": None
}

out_single_conv = dreamy_boi.deep_dream(config)

plt.imshow(out_single_conv)
plt.show()

"""
Testing grad masks below
"""
layers_to_use = [model.Mixed_6c.branch1x1]

grad_mask = np.repeat(np.linspace(0, 1, 512),512*3).reshape(512,512,3).astype(np.float32)**2
grad_mask_2 = np.repeat(np.linspace(1, 0, 512),512*3).reshape(512,512,3).astype(np.float32) 

def custom_func(layer_outputs):
    loss = layer_outputs[0][30].mean()
    return loss

config = {
    "image_path": "images/sample_small.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.1,
    "num_octaves": 11,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": [custom_func],
    "max_rotation": 0.5,
    "grayscale": False,
    "gradient_smoothing_coeff": 0.1,
    "gradient_smoothing_kernel_size": 3,
    "grad_mask": [grad_mask]
}

config2, config3 = config.copy(), config.copy()
config2["grad_mask"] = [grad_mask_2]
config2["custom_func"] = [None]

config3["custom_func"] = [custom_func, None]
config3["grad_mask"] = [grad_mask, grad_mask_2]

out_single_conv_a = dreamy_boi.deep_dream_with_masks(config)
# out_single_conv_b = dreamy_boi.deep_dream_with_masks(config2)
out_single_conv = dreamy_boi.deep_dream_with_masks(config3)

plt.imshow(out_single_conv_a)
plt.show()
plt.imshow(out_single_conv)
plt.show()