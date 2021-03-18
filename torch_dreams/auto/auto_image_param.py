import torch.nn as nn
import torch

from .utils import init_image_param, image_buf_to_rgb

class auto_image_param():
    def __init__(self, height, width, device, standard_deviation):
        self.height = height
        self.width = width
        self.param = init_image_param(height = height, width = width, sd = standard_deviation, device = device)
        self.param.requires_grad_()
        self.optimizer = None

    def fetch_optimizer(self, params_list, optimizer = None, lr = 1e-3, weight_decay = 0.):
        if optimizer is not None:
            optimizer = optimizer(params_list, lr = lr, weight_decay = weight_decay)
        else:
            optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_optimizer(self, lr, weight_decay):

        self.optimizer = self.fetch_optimizer(
                                    params_list = [self.param],
                                    lr = lr,
                                    weight_decay= weight_decay
                                    )
    
    def clip_grads(self, grad_clip = 1.):
        torch.nn.utils.clip_grad_norm_(self.param,grad_clip)

    def to_hwc_tensor(self, device = 'cpu'):
        rgb = image_buf_to_rgb(h = self.height, w = self.width, img_buf = self.param , device= device).permute(1,2,0).to(device)
        return rgb
        
    @property
    def rgb(self):
        rgb = self.to_hwc_tensor().numpy()
        return rgb