import torchvision.transforms as transforms
import torch.nn as nn
import random
from .image_transforms import resize_4d_tensor_by_factor, resize_4d_tensor_by_size

imagenet_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std= [0.229, 0.224, 0.225]
)


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

    def forward(self, tensors = []):
    
        height_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)
        width_factor = random.uniform(a = self.min_size_factor , b = self.max_size_factor)

        outputs = []
        for x in tensors:
            resized_tensor = resize_4d_tensor_by_factor(x = x, height_factor = height_factor, width_factor = width_factor)
            outputs.append(resized_tensor)

        return outputs

class pair_random_affine(nn.Module):
    def __init__(self, degrees, translate_x, translate_y):
        super().__init__()
        self.degrees = degrees
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.affine = transforms.RandomAffine(degrees = self.degrees, translate= (self.translate_x, self.translate_y))

    def forward(self, tensors = []):

        params = self.affine.get_params(degrees = (-self.degrees, self.degrees),  translate= (self.translate_x, self.translate_y), scale_ranges = (1,1), shears = (0,0), img_size = (tensors[0].shape[-2], tensors[0].shape[1]))

        outputs = []
        for x in tensors:
            affined = transforms.functional.affine(x, *params)
            outputs.append(affined)

        return outputs
                