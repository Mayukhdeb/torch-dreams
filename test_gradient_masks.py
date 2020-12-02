import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch_dreams.dreamer import dreamer
import torchvision.models as models
import torch

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)


layers_to_use = [model.Mixed_6c.branch1x1]

grad_mask = np.repeat(np.linspace(0, 1, 512),512*3).reshape(512,512,3).astype(np.float32) 
grad_mask_2 = np.repeat(np.linspace(1, 0, 512),512*3).reshape(512,512,3).astype(np.float32) 

def custom_func(layer_outputs):
    losses = []

    output_channel = layer_outputs[0][30]
    loss_component = torch.nn.MSELoss(reduction='mean')(output_channel, torch.zeros_like(output_channel))
    losses.append(loss_component)

    loss = torch.mean(torch.stack(losses))
    return loss

config = {
    "image_path": "images/sample_small.jpg",
    "layers": layers_to_use,
    "octave_scale": 1.1,
    "num_octaves": 11,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": [custom_func],
    "max_rotation": 0.2,
    "grayscale": False,
    "gradient_smoothing_coeff": 0.5,
    "gradient_smoothing_kernel_size": 9,
    "grad_mask": [grad_mask]
}

config2, config3 = config, config
config2["grad_mask"] = [grad_mask_2]
config2["custom_func"] = [None]

config3["custom_func"] = [custom_func, None]
config3["grad_mask"] = [grad_mask, grad_mask_2]

# print(config3)
out_single_conv_a = dreamy_boi.deep_dream_with_masks(config)
out_single_conv_b = dreamy_boi.deep_dream_with_masks(config2)
out_single_conv = dreamy_boi.deep_dream_with_masks(config3)


plt.imshow(out_single_conv)
plt.show()

