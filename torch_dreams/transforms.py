import torchvision.transforms as transforms
import torch.nn as nn
import random
from .image_transforms import resize_4d_tensor_by_factor, resize_4d_tensor_by_size

class random_resize(nn.Module):
    def __init__(self, max_size_factor, min_size_factor):
        super().__init__()
        self.max_size_factor = max_size_factor
        self.min_size_factor = min_size_factor

    def forward(self, x):
    
        # size = random.randint(a = 300, b = 600)
        # resized= resize_4d_tensor_by_size(x = x, height = size, width = size)

        height_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)
        width_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)

        resized = resize_4d_tensor_by_factor(x = x, height_factor = height_factor, width_factor = width_factor)
        return resized

class pair_random_resize(nn.Module):
    def __init__(self, max_size_factor, min_size_factor):
        super().__init__()
        self.max_size_factor = max_size_factor
        self.min_size_factor = min_size_factor

    def forward(self, x1, x2):
    
        height_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)
        width_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)

        resized_x1 = resize_4d_tensor_by_factor(x = x1, height_factor = height_factor, width_factor = width_factor)
        resized_x2 = resize_4d_tensor_by_factor(x = x2, height_factor = height_factor, width_factor = width_factor)

        return resized_x1,resized_x2

class pair_random_affine(nn.Module):
    def __init__(self, degrees, translate_x, translate_y):
        super().__init__()
        self.degrees = degrees
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.affine = transforms.RandomAffine(degrees = self.degrees, translate= (self.translate_x, self.translate_y))

    def forward(self, x1, x2):

        params = self.affine.get_params(degrees = (-self.degrees, self.degrees),  translate= (self.translate_x, self.translate_y), scale_ranges = (1,1), shears = (0,0), img_size = (x1.shape[-2], x1.shape[1]))

        x1, x2 = transforms.functional.affine(x1, *params), transforms.functional.affine(x2, *params)
        return x1, x2
                