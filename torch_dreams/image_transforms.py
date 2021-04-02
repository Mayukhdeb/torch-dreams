import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


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


def resize_4d_tensor_by_factor(x, height_factor, width_factor):
    res = F.interpolate(x, scale_factor= (height_factor, width_factor), mode = 'bilinear')
    return res

def resize_4d_tensor_by_size(x, height, width):
    res = F.interpolate(x, size =  (height, width), mode = 'bilinear')
    return res