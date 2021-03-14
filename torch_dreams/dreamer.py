import cv2
import tqdm
import torch
import warnings
import numpy as np
from tqdm import tqdm

from. utils import load_image_from_config
from .utils import pytorch_input_adapter
from .utils import pytorch_output_adapter
from .utils import post_process_numpy_image

from .constants import default_config

from .dreamer_utils import default_func_mean
from .dreamer_utils import make_octave_sizes

from .octave_utils import dream_on_octave_with_masks
from .octave_utils import dream_on_octave

from .image_param import image_param

class dreamer():

    """
    Main class definition for torch-dreams:

    model = Any PyTorch deep-learning model
    device = "cuda" or "cpu" depending on GPU availability
    self.config = dictionary containing everything required thats needed for things to work, check the readme (https://github.com/Mayukhdeb/torch-dreams#a-closer-look)
    self.default_func = default loss to be used if no custom_func is defined 
    """

    def __init__(self, model, quiet_mode = False):
        self.model = model
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model moves to GPU if available
        self.model = self.model.to(self.device)
        self.config = default_config.copy()

        self.default_func = default_func_mean
        self.dream_on_octave = dream_on_octave
        self.dream_on_octave_with_masks = dream_on_octave_with_masks
        self.quiet_mode= quiet_mode

        if self.quiet_mode  is False:
            print("dreamer init on: ", self.device)

    def deep_dream(self, config):

        for key in list(config.keys()):
            self.config[key] = config[key]

        image_np = load_image_from_config(self.config)

        original_size = image_np.shape[:-1]

        octave_sizes = make_octave_sizes(
            original_size=original_size, num_octaves= self.config["num_octaves"], octave_scale=self.config["octave_scale"])
    

        image_parameter  = image_param(pytorch_input_adapter(image_np, device = self.device).unsqueeze(0))

        if self.config['add_laplacian'] == True:

            octaves = []
            img = image_param(pytorch_input_adapter(image_np, device = self.device).unsqueeze(0))

            for size in octave_sizes[::-1]:
                old = img.tensor.copy()
                hw = img.tensor.shape[-2], img.tensor.shape[-1]
                img.resize_by_size(height = size[0], width = size[1])
                low = img.tensor.copy()
                img.resize_by_size(height = hw[0], width = hw[1])
                hi = old - img.tensor
                img = low
                octaves.append(hi)

        count = 0        

        for s in tqdm(range(self.config['num_octaves']), disable = self.quiet_mode):
            size = octave_sizes[s]

            image_parameter.resize_by_size(height = size[0], width = size[1])
            image_parameter.tensor.grad = None
            image_parameter.get_optimizer(lr = self.config['lr'])

            if self.config['add_laplacian']:
                if  count > 0:
                    hi = octaves[-count]
                    image_np += hi

            image_parameter = self.dream_on_octave(
                model=self.model,
                image_parameter = image_parameter,
                layers = self.config["layers"],
                iterations = self.config["iterations"],
                lr= self.config["lr"],
                custom_func = self.config["custom_func"],
                max_rotation = self.config["max_rotation"],
                max_roll_x= self.config["max_roll_x"],
                max_roll_y= self.config["max_roll_y"],
                gradient_smoothing_coeff = self.config["gradient_smoothing_coeff"],
                gradient_smoothing_kernel_size= self.config["gradient_smoothing_kernel_size"], 
                default_func=self.default_func, 
                device=self.device
            )
            count += 1

        img_out = image_parameter.tensor.squeeze(0).detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        image_np = post_process_numpy_image(img_out_np)

        return image_np


    def deep_dream_with_masks(self, config):

        for key in list(config.keys()):
            self.config[key] = config[key]

        image_np = load_image_from_config(self.config)

        original_size = image_np.shape[:-1]

        octave_sizes = make_octave_sizes(
            original_size=original_size, num_octaves=self.config["num_octaves"], octave_scale=self.config["octave_scale"])

        image_parameter  = image_param(pytorch_input_adapter(image_np, device = self.device).unsqueeze(0))

        if self.config['add_laplacian'] == True:

            octaves = []
            img = image_param(pytorch_input_adapter(image_np, device = self.device).unsqueeze(0))

            for size in octave_sizes[::-1]:
                old = img.tensor.copy()
                hw = img.tensor.shape[-2], img.tensor.shape[-1]
                img.resize_by_size(height = size[0], width = size[1])
                low = img.tensor.copy()
                img.resize_by_size(height = hw[0], width = hw[1])
                hi = old - img.tensor
                img = low
                octaves.append(hi)

        count = 0

        for s in tqdm(range(self.config['num_octaves']), disable = self.quiet_mode):

            size = octave_sizes[s]

            image_parameter.resize_by_size(height = size[0], width = size[1])
            image_parameter.tensor.grad = None
            # print(image_parameter.tensor.grad)
            image_parameter.get_optimizer(lr = self.config['lr'])

            if self.config['add_laplacian']:
                if  count > 0:
                    hi = octaves[-count]
                    image_np += hi

            if self.config["grad_mask"] is not None:
                grad_mask = [cv2.resize(g, size) for g in self.config["grad_mask"]]

            image_np = self.dream_on_octave_with_masks(
                model=self.model, 
                image_parameter=image_parameter, 
                layers= self.config["layers"], 
                iterations= self.config["iterations"], 
                lr= self.config["lr"], 
                custom_funcs= self.config["custom_func"], 
                max_rotation= self.config["max_rotation"],
                gradient_smoothing_coeff= self.config["gradient_smoothing_coeff"], 
                gradient_smoothing_kernel_size= self.config["gradient_smoothing_kernel_size"], 
                grad_mask= grad_mask, 
                device=self.device, 
                default_func=self.default_func
            )
            count += 1

        img_out = image_parameter.tensor.squeeze(0).detach().cpu()

        img_out_np = img_out.numpy()
        img_out_np = img_out_np.transpose(1,2,0)
        image_np = post_process_numpy_image(img_out_np)

        return image_np
