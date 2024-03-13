import torch

from .utils import init_series_param
from .utils import fft_to_series


class BaseSeriesParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = None
        self.length = None
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
        """Moves the series from the frequency domain to Spatial (Visible to the eyes)

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

    def fetch_optimizer(self, params_list, optimizer=None, lr=1e-3, weight_decay=0.0):
        if optimizer is not None:
            optimizer = optimizer(params_list, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=weight_decay)
        return optimizer

    def get_optimizer(self, lr, weight_decay):
        self.optimizer = self.fetch_optimizer(
            params_list=[self.param], lr=lr, weight_decay=weight_decay
        )

    def clip_grads(self, grad_clip=1.0):
        torch.nn.utils.clip_grad_norm_(self.param, grad_clip)

    def to_cl_tensor(self, device="cpu"):
        """Return CL series tensor (channels, length).

        Args:
            device (str):  The device to operate on ('cpu' or 'cuda').

        Returns:
            torch.Tensor
        """
        t = self.forward(device=device)[0].detach()
        return t

    def to_lc_tensor(self, device="cpu"):
        """Return LC series tensor (length, channels).

        Args:
            device (str):  The device to operate on ('cpu' or 'cuda').

        Returns:
            torch.Tensor
        """
        t = self.forward(device=device)[0].permute(1, 0).detach()
        return t

    def __array__(self):
        """Generally used for plt.imshow(), converts the series parameter to a NCL numpy array

        Returns:
            numpy.ndarray
        """
        return self.to_cl_tensor().numpy()

    def save(self, filename):
        """Save an image_param as an image. Uses PIL to save the image

        usage:

            image_param.save(filename = 'my_image.jpg')

        Args:
            filename (str): image.jpg
        """
        tensor = self.to_cl_tensor()
        torch.save(tensor, filename)


class AutoSeriesParam(BaseSeriesParam):
    """Trainable series parameter which can be used to activate
       different parts of a neural net

    Args:
        length (int): The sequence length of the series
        channels (int): The number of channels of the series

        device (str): 'cpu' or 'cuda'
        standard_deviation (float): Standard deviation of the series initiated
         in the frequency domain.
        batch_size (int): The batch size of the input tensor. If batch_size=1,
         no batch dimension is expected.
    """

    def __init__(self, length, channels, device, standard_deviation, batch_size: int = 1):
        super().__init__()

        self.length = length
        self.channels = channels
        self.standard_deviation = standard_deviation
        self.batch_size = batch_size
        self.device = device

        self.optimizer = None

        """
        odd length is resized to even with one extra element
        """
        if self.length % 2 == 1:
            self.param = init_series_param(
                channels=self.channels,
                length=self.length + 1,
                sd=standard_deviation,
                device=device,
            )
        else:
            self.param = init_series_param(
                channels=self.channels,
                length=self.length,
                sd=standard_deviation,
                device=device,
            )
        self.param.requires_grad_()

    def postprocess(self, device):
        series = fft_to_series(
            channels=self.channels,
            length=self.length,
            series_parameter=self.param,
            device=device,
        )
        #TODO: img = lucid_colorspace_to_rgb(t=img, device=device)
        series = torch.sigmoid(series)
        return series

    def normalize(self,x, device):
        # TODO: implement normalization
        #return normalize(x = x, device= device)
        return x

    def forward(self, device):
        # TODO: add normalization
        if self.batch_size == 1:
            return self.normalize(self.postprocess(device=device), device=device)
        else:
            return torch.cat(
                [
                    self.normalize(self.postprocess(device=device), device=device)
                    for i in range(self.batch_size)
                ],
                dim=0,
            )
