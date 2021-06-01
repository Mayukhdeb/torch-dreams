import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torch.nn as nn


import warnings
warnings.filterwarnings("ignore")

transform_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

transform_and_rotate = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees = 5, resample=False, expand=False, center=None, fill=None),
                transforms.ToTensor()
            ])

class InverseTransform(nn.Module):
    def __init__(self, old_mean, old_std, new_transforms):
        super().__init__()
        self.inverse_transform =  transforms.Compose([ 
            transforms.Normalize(
                mean = [ 0., 0., 0. ],
                std = [ 1/old_std[0], 1/old_std[1], 1/old_std[2]]
                ),
            transforms.Normalize(
                mean = [ -old_mean[0], -old_mean[1], -old_mean[2]],
                std = [ 1., 1., 1. ]
                ),
        ]) 

        self.new_transform =  new_transforms
        
    def forward(self, x):
        x=  self.inverse_transform(x)
        x = self.new_transform(x)
        return x

    def __repr__(self):
        return 'InverseTransform(\n        ' + str(self.inverse_transform) + '\n        ' + str(self.new_transform) + ')'


def resize_4d_tensor_by_factor(x, height_factor, width_factor):
    res = F.interpolate(x, scale_factor= (height_factor, width_factor), mode = 'bilinear')
    return res

def resize_4d_tensor_by_size(x, height, width):
    res = F.interpolate(x, size =  (height, width), mode = 'bilinear')
    return res