import cv2
import tqdm
import torch
import numpy as np
from tqdm import tqdm

from .utils import load_image
from .utils import preprocess_numpy_img
from .utils import pytorch_input_adapter
from .utils import pytorch_output_adapter
from .utils import post_process_numpy_image

from .constants import default_config
from .dreamer_utils import default_func_norm
from .dreamer_utils import make_octave_sizes

from .octave_utils import dream_on_octave_with_masks
from .octave_utils import dream_on_octave

class dreamer():

    """
    Main class definition for torch-dreams:

    model = Any PyTorch deep-learning model
    preprocess_func = Set of torch transforms required for the model wrapped into a function. See torch_dreams.utils for examples
    deprocess_func <optional> = set of reverse transforms, to be applied before converting the image back to numpy
    device = checks for a GPU, uses the GPU for tensor operations if available
    """

    def __init__(self, model):
        self.model = model
        self.model = self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model moves to GPU if available
        self.model = self.model.to(self.device)
        self.config = default_config.copy()

        self.default_func = default_func_norm
        self.dream_on_octave = dream_on_octave
        self.dream_on_octave_with_masks = dream_on_octave_with_masks

        print("dreamer init on: ", self.device)

    def deep_dream(self, config):

        for key in list(config.keys()):
            self.config[key] = config[key]

        image_path = self.config["image_path"]
        grayscale = self.config["grayscale"]
        
        original_image = load_image(image_path, grayscale=grayscale)
        image_np = preprocess_numpy_img(original_image, grayscale=grayscale)
        if grayscale is True:
            image_np = np.expand_dims(image_np, axis=-1)

        original_size = image_np.shape[:-1]

        octave_sizes = make_octave_sizes(
            original_size=original_size, num_octaves= self.config["num_octaves"], octave_scale=self.config["octave_scale"])

        """
        source: 
        https://github.com/ProGamerGov/Protobuf-Dreamer/blob/bb9943411129127220c131793264c8b24a71a6c0/pb_dreamer.py#L105
        """
    
        img = original_image.copy()
        octaves = []  

        for size in octave_sizes[::-1]:
            hw = img.shape[1], img.shape[0]
            lo = cv2.resize(img, size)
            hi = img- cv2.resize(lo, hw)
            img = lo
            octaves.append(hi)

        count = 0
        for new_size in tqdm(octave_sizes):

            image_np = cv2.resize(image_np, new_size)

            if grayscale is True:
                image_np = np.expand_dims(image_np, axis=-1)

            if  count > 0:
                hi = octaves[-count]
                image_np += hi

            image_np = self.dream_on_octave(
                model=self.model,
                image_np = image_np,
                layers = self.config["layers"],
                iterations = self.config["iterations"],
                lr= self.config["lr"],
                custom_func = self.config["custom_func"],
                max_rotation = self.config["max_rotation"],
                gradient_smoothing_coeff = self.config["gradient_smoothing_coeff"],
                gradient_smoothing_kernel_size= self.config["gradient_smoothing_kernel_size"], 
                grayscale=self.config["grayscale"], 
                default_func=self.default_func, 
                device=self.device
            )
            count += 1

        image_np = post_process_numpy_image(image_np)
        return image_np


    def deep_dream_with_masks(self, config):

        for key in list(config.keys()):
            self.config[key] = config[key]

        image_path = self.config["image_path"]
        grayscale = self.config["grayscale"]

        original_image = load_image(self.config["image_path"], grayscale=grayscale)
        image_np = preprocess_numpy_img(original_image, grayscale=grayscale)

        if grayscale is True:
            image_np = np.expand_dims(image_np, axis=-1)

        original_size = image_np.shape[:-1]

        octave_sizes = make_octave_sizes(
            original_size=original_size, num_octaves=self.config["num_octaves"], octave_scale=self.config["octave_scale"])

        img = original_image.copy()
        octaves = []  

        for size in octave_sizes[::-1]:
            hw = img.shape[1], img.shape[0]
            lo = cv2.resize(img, size)
            hi = img- cv2.resize(lo, hw)
            img = lo
            octaves.append(hi)

        count = 0

        for new_size in tqdm(octave_sizes):

            image_np = cv2.resize(image_np, new_size)

            if grayscale is True:
                image_np = np.expand_dims(image_np, axis=-1)

            if  count > 0:
                hi = octaves[-count]
                image_np += hi

            if self.config["grad_mask"] is not None:
                grad_mask = [cv2.resize(g, new_size) for g in self.config["grad_mask"]]

            image_np = self.dream_on_octave_with_masks(
                model=self.model, 
                image_np=image_np, 
                layers= self.config["layers"], 
                iterations= self.config["iterations"], 
                lr= self.config["lr"], 
                custom_funcs= self.config["custom_func"], 
                max_rotation= self.config["max_rotation"],
                gradient_smoothing_coeff= self.config["gradient_smoothing_coeff"], 
                gradient_smoothing_kernel_size= self.config["gradient_smoothing_kernel_size"], 
                grad_mask= grad_mask, 
                grayscale=grayscale, 
                device=self.device, 
                default_func=self.default_func
            )
            count += 1

        image_np = post_process_numpy_image(image_np)
        return image_np
