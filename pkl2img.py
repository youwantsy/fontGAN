import torch
from torchvision.utils import save_image


a = torch.load('/home/jsy/font/fixed_set/t_fixed_target2.pkl')
save_image((a+1)/2*255, '/home/jsy/font/fixed_set/t_fixed_target2.png', pad_value=255)