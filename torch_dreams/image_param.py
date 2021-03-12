import torch.nn as nn
import torch.optim as optim

import torch
from .utils import rotate_image_tensor
from .utils import roll_torch_tensor

class image_param():
    def __init__(self, image_tensor):
        self.tensor = image_tensor
        self.tensor.requires_grad = True

    def get_optimizer(self, lr, optimizer = None, momentum = 0., weight_decay = 0.):
        if optimizer is None:
            self.optimizer = optim.SGD([self.tensor], lr = lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer([self.tensor], lr = lr, weight_decay = weight_decay)
        
    def set_gradients(self, gradients_tensor):
        """Sets the gradients of the tensor 

        Args:
            gradients_tensor (torch.tensor): 4D tensor of shape [batch, C, H, W]
        """
        self.tensor.grad = None
        self.tensor.grad = gradients_tensor
    
    def clip_to_bounds(self, upper, lower):
        self.tensor = torch.max(torch.min(self.tensor, upper), lower)

    def roll(self, roll_x, roll_y):
        return roll_torch_tensor(self.tensor.clone(), roll_x= roll_x, roll_y = roll_y)
    