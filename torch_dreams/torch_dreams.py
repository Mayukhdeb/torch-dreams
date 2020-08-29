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



image_main = cv2.imread("sample_images/cloudy-mountains.jpg")
image_sample = cv2.cvtColor(image_main, cv2.COLOR_BGR2RGB)
image_sample = cv2.resize(image_sample, (1024,1024))

plt.imshow(image_sample)
plt.show()

model = models.vgg19(pretrained=True)
layers = list(model.features.children())
model.eval()


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])




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


#Function to make gradients calculations from the output channels of the target layer.
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

#Function to run the dream. The excesive casts to and from numpy arrays is to make use of the np.roll() function.
#By rolling the image randomly everytime the gradients are computed, we prevent a tile effect artifact from appearing.
def dream(image, net, layer, iterations, lr, out_channels = None):
    image_numpy = np.array(image)
    image_tensor = transforms.ToTensor()(image_numpy)
    image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
    denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                                ])
    for i in range(iterations):
        roll_x = np.random.randint(image_numpy.shape[0])
        roll_y = np.random.randint(image_numpy.shape[1])
        img_roll = np.roll(np.roll(image_tensor.detach().cpu().numpy().transpose(1,2,0), roll_y, 0), roll_x, 1)
        img_roll_tensor = torch.tensor(img_roll.transpose(2,0,1), dtype = torch.float)
        gradients_np = get_gradients(img_roll_tensor, net, layer, out_channels).detach().cpu().numpy()
        gradients_np = np.roll(np.roll(gradients_np, -roll_y, 1), -roll_x, 2)
        gradients_tensor = torch.tensor(gradients_np)
        image_tensor.data = image_tensor.data + lr * gradients_tensor.data

    img_out = image_tensor.detach().cpu()
    img_out = denorm(img_out)
    img_out_np = img_out.numpy()
    img_out_np = img_out_np.transpose(1,2,0)
    img_out_np = np.clip(img_out_np, 0, 1)
    
    return img_out_np

def deep_dream(image_np, model, layer_index, octave_scale, num_octaves, iterations, lr):
    original_size = image_np.shape[:2]
    layer = list( model.features.modules() )[layer_index]

    for n in tqdm(range(-num_octaves, 1)):
        octave_size = tuple( np.array(original_size) * octave_scale**n)
        new_size = (int(octave_size[1]), int(octave_size[0]))

        image_np = cv2.resize(image_np, new_size)

        image_np = dream(image_np, model, layer, iterations = iterations, lr = lr, out_channels = None)

    return image_np

dreamed = deep_dream(image_sample, model,19, 1.5, 2, 2, 0.21 )

cv2.imwrite('dream.jpg', dreamed)