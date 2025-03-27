import torch


class BaseSeriesParam(torch.nn.Module):
    def __init__(self, batch_size, channels, length, param, normalize_mean, normalize_std, device):
        super().__init__()

        self.batch_size = batch_size
        self.channels = channels
        self.length = length

        if normalize_mean is None:
            normalize_mean = torch.FloatTensor([0] * channels)
        self.normalize_mean = normalize_mean

        if normalize_std is None:
            normalize_std=torch.FloatTensor([1] * channels)
        self.normalize_std = normalize_std

        self.param = param
        self.param.requires_grad_()

        self.device = device

        self.optimizer = None

    def forward(self, device):
        """This is what the model gets, should be processed and normalized with the right values

        The model gets:  self.normalize(self.postprocess(self.param))

        Raises:
            NotImplementedError: Implemented below, you're in the base class.
        """

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

    def postprocess(self):
        """Moves the series from the frequency domain to Spatial (Visible to the eyes)

        Raises:
            NotImplementedError: Implemented below, you're in the base class.
        """
        raise NotImplementedError

    def normalize(self, x, device='cuda'):
        """Normalizing wrapper"""
        return (
                (x - self.normalize_mean[..., None].to(device))
                / self.normalize_std[..., None].to(device)
        )

    def denormalize(self, x, device='cuda'):
        """Denormalizing wrapper."""
        return (
                x * self.normalize_std[..., None].to(device)
                + self.normalize_mean[..., None].to(device)
        )

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
        return torch.nn.utils.clip_grad_norm_(self.param, grad_clip)

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
