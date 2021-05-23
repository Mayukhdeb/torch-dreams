from .custom_image_param import custom_image_param

class masked_image_param(custom_image_param):
    def __init__(self, image, mask, device):
        super().__init__(image = image, device= device)

        self.mask = mask.to(self.device)

        assert self.mask.shape == self.to_nchw_tensor(device = self.device).shape

    def forward(self, device):
        return self.normalize(self.postprocess(device = device), device= device).clamp(0,1) * self.mask
