import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn


import warnings

warnings.filterwarnings("ignore")

transform_to_tensor = transforms.Compose([transforms.ToTensor()])

transform_and_rotate = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomRotation(
            degrees=5, expand=False, center=None, fill=None
        ),
        transforms.ToTensor(),
    ]
)

def unnormalize_image_tensor(img, mean, std):
    img = img * std.view(1,3,1,1).cuda() + mean.view(1,3,1,1).cuda()
    return img.clamp(0,1)


class InverseTransform(nn.Module):
    def __init__(self, old_mean, old_std, new_transforms):
        super().__init__()
        self.old_mean = old_mean
        self.old_std = old_std
        self.new_transform = new_transforms

    def forward(self, x):
        x = unnormalize_image_tensor(
            img = x,
            mean = self.old_mean,
            std = self.old_std
        )
        x = self.new_transform(x)
        return x

    def __repr__(self):
        return (
            "InverseTransform(\n        "
            + str(self.inverse_transform)
            + "\n        "
            + str(self.new_transform)
            + ")"
        )


def resize_4d_tensor_by_factor(x, height_factor, width_factor):
    res = F.interpolate(x, scale_factor=(height_factor, width_factor), mode="bilinear")
    return res


def resize_4d_tensor_by_size(x, height, width):
    res = F.interpolate(x, size=(height, width), mode="bilinear")
    return res
