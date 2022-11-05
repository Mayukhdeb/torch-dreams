import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torchvision.transforms as transforms

from .transforms import random_resize, pair_random_resize, pair_random_affine

from .constants import Constants
from .losses import CaricatureLoss
from .image_transforms import InverseTransform
from .auto_image_param import AutoImageParam
from .dreamer_utils import Hook, default_func_mean
from .masked_image_param import MaskedImageParam
from .batched_image_param import BatchedImageParam


class Dreamer:
    """wrapper over a pytorch model for feature visualization

    Args:
        model (torch.nn.Module): pytorch model
        quiet (bool, optional): enable or disable progress bar. Defaults to True.
        device (str, optional): 'cpu' or 'cuda'. Defaults to 'cuda'.

    Example:

    ```python
    import torchvision.models as models
    from torch_dreams import Dreamer

    model = models.inception_v3(pretrained=True)
    dreamy_boi = Dreamer(model, device = 'cuda')
    ```
    """

    def __init__(self, model, quiet=False, device="cuda"):

        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.default_func = default_func_mean
        self.transforms = None
        self.quiet = quiet
        self.__custom_normalization_transform__ = None

    def get_default_transforms(
        self, rotate, scale_max, scale_min, translate_x, translate_y
    ):
        self.transforms = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=rotate, translate=(translate_x, translate_y)
                ),
                random_resize(max_size_factor=scale_max, min_size_factor=scale_min),
            ]
        )

        if self.__custom_normalization_transform__ is not None:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=rotate, translate=(translate_x, translate_y)
                    ),
                    random_resize(max_size_factor=scale_max, min_size_factor=scale_min),
                    self.__custom_normalization_transform__,
                ]
            )

    def set_custom_normalization(self, normalization_transform):
        """Overrides the usual imagenet normalization to be compatible with any other normalization defined by the user.

        Args:
            normalization_transform (torch.nn.Module): transforms that would normalize the image as per your model.

        Example:

        ```python
        t = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        dreamer.set_custom_normalization(normalization_transform = t)
        ```
        """

        assert isinstance(normalization_transform, nn.Module) == True, (
            "normalization_transform should be an instance of torch.nn.module, not: "
            + str(type(normalization_transform))
        )

        self.__custom_normalization_transform__ = InverseTransform(
            old_mean=Constants.imagenet_mean,
            old_std=Constants.imagenet_std,
            new_transforms=normalization_transform,
        )

    def set_custom_transforms(self, transforms):
        """Sets a custom set of transforms as the robustness transforms to be used for training the image parameter. Does not override custom image normalization if configured.

        Args:
            transforms (torch.nn.Module): Bunch of transforms to be used for training the image parameter. Generally `torchvision.transforms.Compose([your_transforms])` is preferred.

        Example:

        ```python
        my_transforms = transforms.Compose([
            transforms.RandomAffine(degrees = 10, translate = (0.5,0.5)),
            transforms.RandomHorizontalFlip(p = 0.3)
        ])

        dreamy_boi.set_custom_transforms(transforms = my_transforms)
        ```
        """
        if self.__custom_normalization_transform__ == None:
            self.transforms = transforms
        else:
            self.transforms = torchvision.transforms.Compose(
                [transforms, self.__custom_normalization_transform__]
            )

    def render(
        self,
        layers,
        image_parameter=None,
        width=256,
        height=256,
        iters=120,
        lr=9e-3,
        rotate_degrees=15,
        scale_max=1.2,
        scale_min=0.5,
        translate_x=0.0,
        translate_y=0.0,
        custom_func=None,
        weight_decay=0.0,
        grad_clip=1.0,
    ):
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
            image_parameter instance: To show image, use: plt.imshow(image_parameter)
        """
        if image_parameter is None:
            image_parameter = AutoImageParam(
                height=height, width=width, device=self.device, standard_deviation=0.01
            )
        else:
            image_parameter = deepcopy(image_parameter)

        if image_parameter.optimizer is None:
            image_parameter.get_optimizer(lr=lr, weight_decay=weight_decay)

        if self.transforms is None:
            self.get_default_transforms(
                rotate=rotate_degrees,
                scale_max=scale_max,
                scale_min=scale_min,
                translate_x=translate_x,
                translate_y=translate_y,
            )

        hooks = []
        for layer in layers:
            hook = Hook(layer)
            hooks.append(hook)

        if isinstance(image_parameter, MaskedImageParam):
            self.random_resize_pair = pair_random_resize(
                max_size_factor=scale_max, min_size_factor=scale_min
            )
            self.random_affine_pair = pair_random_affine(
                degrees=rotate_degrees, translate_x=translate_x, translate_y=translate_y
            )

        for i in tqdm(range(iters), disable=self.quiet):

            image_parameter.optimizer.zero_grad()

            img = image_parameter.forward(device=self.device)

            if isinstance(image_parameter, MaskedImageParam):
                (
                    img_transformed,
                    mask_transformed,
                    original_image_transformed,
                ) = self.random_resize_pair(
                    tensors=[
                        img,
                        image_parameter.mask.to(self.device),
                        image_parameter.original_nchw_image_tensor,
                    ]
                )
                (
                    img_transformed,
                    mask_transformed,
                    original_image_transformed,
                ) = self.random_affine_pair(
                    [img_transformed, mask_transformed, original_image_transformed]
                )

                img = img_transformed * mask_transformed.to(
                    self.device
                ) + original_image_transformed.float() * (
                    1 - mask_transformed.to(self.device)
                )

            else:
                img = self.transforms(img)

            model_out = self.model(img)

            layer_outputs = []

            for hook in hooks:
                ## if it's a BatchedImageParam, then include all batch items from hook output
                if isinstance(image_parameter, BatchedImageParam):
                    out = hook.output
                else:
                    ## else select only the first and only batch item
                    out = hook.output[0]

                layer_outputs.append(out)

            if custom_func is not None:
                loss = custom_func(layer_outputs)
            else:
                loss = self.default_func(layer_outputs)
            loss.backward()
            image_parameter.clip_grads(grad_clip=grad_clip)
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

    def caricature(
        self,
        input_tensor,
        layers,
        power=1.0,
        image_parameter=None,
        iters=120,
        lr=9e-3,
        rotate_degrees=15,
        scale_max=1.2,
        scale_min=0.5,
        translate_x=0.1,
        translate_y=0.1,
        weight_decay=1e-3,
        grad_clip=1.0,
        static=False,
    ):
        """Generates an "exaggerated" reconstruction of the input image by
        optimizing random noise to replicate and "exaggerate" the activations of
        certain layers after feeding the input image

        Args:
            input_tensor (torch.tensor): input image tensor, should be clipped between 0,1 and Normalised w.r.t image mean and std
            layers (list): list of layers whose activations are to be registered and later replicated on random noise
            power (float, optional): Determined the "exaggeration" factor. 1 means "no exaggeration". Defaults to 1..
            image_parameter ([type], optional): [description]. Defaults to None.
            iters (int, optional): Number of optimization steps. Defaults to 120.
            lr (float, optional): learning rate. Defaults to 9e-3.
            rotate_degrees (int, optional): Maximum amount of random rotation in degrees. Defaults to 15.
            scale_max (float, optional): Maximum scale factor for random scaling. Defaults to 1.2.
            scale_min (float, optional): Minimum scale factor for random scaling. Defaults to 0.5.
            translate_x (float, optional): Maximum amount of horizontal jitter. Defaults to 0.1.
            translate_y (float, optional): Maximum amount of vertical jitter. Defaults to 0.1.
            weight_decay (float, optional): Helps reduce high frequency noise, higher means lower noise. Defaults to 1e-3.
            grad_clip (float, optional): Maximum value of grad norm. Smaller -> more "careful" steps. Defaults to 1..

        Example:
        ```python
        param = dreamy_boi.caricature(
            input_tensor = image_tensor,
            layers = [model.some_layer],
            power= 1.0
        )
        param.save('caricature.jpg')
        ```

        Returns:
            instance of torch_dreams.auto_image_param.BaseImageParam
        """

        if image_parameter is None:
            height, width = input_tensor.shape[-2], input_tensor.shape[-1]
            image_parameter = AutoImageParam(
                height=height, width=width, device=self.device, standard_deviation=0.01
            )
        else:
            image_parameter = deepcopy(image_parameter)

        if image_parameter.optimizer is None:
            image_parameter.get_optimizer(lr=lr, weight_decay=weight_decay)

        hooks = []
        for layer in layers:
            hook = Hook(layer)
            hooks.append(hook)

        caricature_loss = CaricatureLoss(power=power)

        if static == False:
            """
            if static is false, we have to transforms the input tensor and the image parameter identically before each iteration
            """
            self.random_resize_pair = pair_random_resize(
                max_size_factor=scale_max, min_size_factor=scale_min
            )
            self.random_affine_pair = pair_random_affine(
                degrees=rotate_degrees, translate_x=translate_x, translate_y=translate_y
            )
        else:
            """
            if static is true, then we collect the layer outputs first, and use the same ideal layer outputs as the target at each iteration,
            this would be useful particularly for adversarial examples where the outcome depends on the position of each pixel
            """
            self.get_default_transforms(
                rotate=rotate_degrees,
                scale_max=scale_max,
                scale_min=scale_min,
                translate_x=translate_x,
                translate_y=translate_y,
            )
            model_out = self.model(input_tensor.to(self.device))

            ideal_layer_outputs = []

            for hook in hooks:
                out = hook.output[0]
                ideal_layer_outputs.append(out)

        for i in tqdm(range(iters), disable=self.quiet):

            image_parameter.optimizer.zero_grad()

            img = image_parameter.forward(device=self.device)

            if static == False:
                img_transformed, input_tensor_transformed = self.random_resize_pair(
                    tensors=[img, input_tensor]
                )
                img_transformed, input_tensor_transformed = self.random_affine_pair(
                    tensors=[img_transformed, input_tensor_transformed]
                )
            else:
                img_transformed = self.transforms(img)

            model_out = self.model(img_transformed)

            layer_outputs = []

            for hook in hooks:
                out = hook.output[0]
                layer_outputs.append(out)

            if static == False:
                model_out = self.model(input_tensor_transformed.to(self.device))

                ideal_layer_outputs = []

                for hook in hooks:
                    out = hook.output[0]
                    ideal_layer_outputs.append(out)

            loss = caricature_loss.forward(
                layer_outputs=layer_outputs, ideal_layer_outputs=ideal_layer_outputs
            )

            loss.backward()
            image_parameter.clip_grads(grad_clip=grad_clip)
            image_parameter.optimizer.step()

        for hook in hooks:
            hook.close()

        return image_parameter
