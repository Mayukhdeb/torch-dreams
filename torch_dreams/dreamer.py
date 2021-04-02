import torch 
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as transforms

from .utils import (
    fft_to_rgb, 
    lucid_colorspace_to_rgb, 
    normalize, 
)

from .transforms import random_resize
from .auto_image_param import auto_image_param
from .dreamer_utils import Hook, default_func_mean

class dreamer():
    """wrapper over a pytorch model for feature visualization

        Args:
            model (torch.nn.Module): pytorch model 
            quiet (bool, optional): enable or disable progress bar. Defaults to True.
            device (str, optional): 'cpu' or 'cuda'. Defaults to 'cuda'.
    """
    def __init__(self, model, quiet = False, device = 'cuda'):
        
        self.model = model 
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.default_func = default_func_mean
        self.transforms = None
        self.quiet = quiet

    def get_default_transforms(self, rotate, scale_max, scale_min, translate_x, translate_y):
        self.transforms= transforms.Compose([
            transforms.RandomAffine(degrees = rotate, translate= (translate_x, translate_y)),
            random_resize(max_size_factor = scale_max, min_size_factor = scale_min),
        ])

    def set_custom_transforms(self, transforms):
        self.transforms = transforms

    def render(self, layers, image_parameter = None,  width= 256, height = 256, iters = 120, lr = 9e-3, rotate_degrees = 15,  scale_max = 1.2,  scale_min = 0.5, translate_x = 0., translate_y = 0.,  custom_func = None, weight_decay = 0., grad_clip = 1.):
        """core function to visualize elements form within the pytorch model

        WARNING: width and height would be ignored if image_parameter is not None

        Args:
            layers (iterable): [model.layer1, model.layer2...]
            image_parameter: instance of torch_dreams.auto.auto_image_param
            width (int): width of image to be optimized 
            height (int): height of image to be optimized 
            iters (int): number of iterations, higher -> stronger visualization
            lr (float): learning rate
            rotate_degrees (int): max rotation in default transforms
            scale_max (float, optional): Max image size factor. Defaults to 1.1.
            scale_min (float, optional): Minimum image size factor. Defaults to 0.5.
            translate_x (float, optional): Maximum translation factor in x direction
            translate_y (float, optional): Maximum translation factor in y direction
            custom_func (function, optional): See docs for a better explanation. Defaults to None.
            weight_decay (float, optional): Weight decay for default optimizer. Helps prevent high frequency noise. Defaults to 0.
            grad_clip (float, optional): Maximum value of norm of gradient. Defaults to 1.

        Returns:
            image_parameter instance: To show image, use: plt.imshow(image_parameter.rgb)
        """
        if image_parameter is None:
            image_parameter = auto_image_param(height= height, width = width, device = self.device, standard_deviation = 0.01)
        else:
            image_parameter = deepcopy(image_parameter)

        if image_parameter.optimizer is None:
            image_parameter.get_optimizer(lr = lr, weight_decay = weight_decay)

        if self.transforms is None:
            self.get_default_transforms(rotate = rotate_degrees, scale_max = scale_max, scale_min= scale_min, translate_x = translate_x, translate_y = translate_y)

        hooks = []
        for layer in layers:
            hook = Hook(layer)
            hooks.append(hook)

        for i in tqdm(range(iters), disable= self.quiet):
            image_parameter.optimizer.zero_grad()

            img = fft_to_rgb(height = image_parameter.height, width =  image_parameter.width, image_parameter =  image_parameter.param, device= self.device)

            img = lucid_colorspace_to_rgb(img,device= self.device)
            img = torch.sigmoid(img)
            img = normalize(img, device= self.device)
            img = self.transforms(img)

            model_out = self.model(img)

            layer_outputs = []

            for hook in hooks:
                out = hook.output[0]
                layer_outputs.append(out)

            if custom_func is not None:
                loss = custom_func(layer_outputs)
            else:
                loss = self.default_func(layer_outputs)
            loss.backward()
            image_parameter.clip_grads(grad_clip= grad_clip)
            image_parameter.optimizer.step()
        

        for hook in hooks:
            hook.close()

        return image_parameter
    
    def get_snapshot(self, layers, input_tensor):
        """Registers the outputs of a set of layers within a model

        Args:
            layers (list): [model.layer1, model.layer2,...]
            input_tensor (torch.Tensor): NCHW tensor to be fed into the model

        Returns:
            [list]: list of layer outputs
        """

        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)

            hooks = []
            for layer in layers:
                hook = Hook(layer)
                hooks.append(hook)


            model_out = self.model(input_tensor.float())

            layer_outputs = []

            for hook in hooks:
                out = hook.output[0]
                layer_outputs.append(out)

        return layer_outputs