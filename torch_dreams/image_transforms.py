import torchvision.transforms as transforms
import torch
import torch.nn.functional as F


import warnings
warnings.filterwarnings("ignore")

transform_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

transform_and_rotate = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees = 5, resample=False, expand=False, center=None, fill=None),
                transforms.ToTensor()
            ])

def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype = torch.float32, device = 'cpu'):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype).to(device)
    x = F.grid_sample(x, grid, align_corners= True)
    return x