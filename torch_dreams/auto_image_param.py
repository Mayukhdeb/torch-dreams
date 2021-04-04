import torch.nn as nn
import torch
import numpy as np

from .utils import init_image_param
import torchvision.transforms as transforms

from .utils import fft_to_rgb, lucid_colorspace_to_rgb, normalize

class BaseImageParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.height = None
        self.width = None
        self.param = None
        self.optimizer = None
    
    def forward(self):
        """This is what the model gets, should be processed and normalized with the right values

        The model gets:  self.normalize(self.postprocess(self.param))

        Raises:
            NotImplementedError: Implemented below, you're in the base class.
        """
        raise NotImplementedError

    def postprocess(self):
        """Moves the image from the frequency domain to Spatial (Visible to the eyes)

        Raises:
            NotImplementedError: Implemented below, you're in the base class.
        """
        raise NotImplementedError

    def normalize(self):
        """Normalizing wrapper, you can either use torchvision.transforms.Normalize() or something else

        Raises:
            NotImplementedError: Implemented below, you're in the base class.
        """
        raise NotImplementedError
        
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
        rgb = self.postprocess(device = device)[0].permute(1,2,0).detach()
        return rgb

    def to_chw_tensor(self, device = 'cpu'):
        t = self.postprocess(device = device)[0].detach()
        return t

    def __array__(self):
        """Generally used for plt.imshow(), converts the image parameter to a numpy array

        Returns:
            numpy.ndarray
        """
        return self.to_hwc_tensor().numpy()

    def save(self, filename):
        """Save an image_param as an image. Uses PIL to save the image

        usage:
            
            image_param.save(filename = 'my_image.jpg')
            
        Args:
            filename (str): image.jpg
        """
        ten = self.to_chw_tensor()
        im = transforms.ToPILImage()(ten)
        im.save(filename)


class auto_image_param(BaseImageParam):
    """Trainable image parameter which can be used to activate 
           different parts of a neural net

        Args:
            height (int): Height of image
            width (int): Width of image
            device (str): 'cpu' or 'cuda'
            standard_deviation (float): Standard deviation of the image initiated in the frequency domain. ). 0.01 works well
    """
    def __init__(self, height, width, device, standard_deviation):
        
        super().__init__()
        self.height = height
        self.width = width

        '''
        odd width is resized to even with one extra column
        '''
        if self.width %2 ==1:
            self.param = init_image_param(height = self.height, width = width + 1, sd = standard_deviation, device = device)
        else:
            self.param = init_image_param(height = self.height, width = self.width, sd = standard_deviation, device = device)

        self.param.requires_grad_()
        self.optimizer = None

    def postprocess(self, device):
        img = fft_to_rgb(height = self.height, width = self.width,  image_parameter = self.param, device= device)
        img = lucid_colorspace_to_rgb(t = img, device = device)
        img = torch.sigmoid(img)
        return img

    def normalize(self,x, device):
        return normalize(x = x, device= device)

    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device)
