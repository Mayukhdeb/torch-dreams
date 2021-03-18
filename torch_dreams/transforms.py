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