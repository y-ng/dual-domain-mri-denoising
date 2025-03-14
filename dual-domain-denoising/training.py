import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fastmri
from fastmri.data import transforms
from constants import *
from models import UNet_kdata, UNet_image

np.random.seed(SEED)

# function to convert kspace to image space
def kspace_to_image(kdata):
    kspace_tensor = transforms.to_tensor(kdata)
    # inverse fourier to get complex img 
    image_data = fastmri.ifft2c(kspace_tensor)
    # absolute value of img
    image_abs = fastmri.complex_abs(image_data)

    return image_abs


def main():
    # load pkl data
    print('Read in data from files...')
    with open(NOISY_KDATA_PATH, 'rb') as handle:
        noisy_kdata = pk.load(handle)
        print(noisy_kdata.dtype, noisy_kdata.shape)

    with open(CLEAN_KDATA_PATH, 'rb') as handle:
        clean_kdata = pk.load(handle)
        print(clean_kdata.dtype, clean_kdata.shape)

    kdata_train_loader = DataLoader(torch.tensor(noisy_kdata), torch.tensor(clean_kdata))

    with open(NOISY_IMAGE_PATH, 'rb') as handle:
        noisy_image = pk.load(handle)
        print(noisy_image.dtype, noisy_image.shape)

    with open(CLEAN_IMAGE_PATH, 'rb') as handle:
        clean_image = pk.load(handle)
        print(clean_image.dtype, clean_image.shape)

    image_train_loader = DataLoader(torch.tensor(noisy_image), torch.tensor(clean_image))

    u_k_net = UNet_kdata()
    u_i_net = UNet_image()

    # 25 total epochs
    batch_size = 4
    u_k_optimizer = torch.optim.Adam(u_k_net.parameters(), lr=1e-4)
    u_i_optimizer = torch.optim.Adam(u_i_net.parameters(), lr=1e-4)
    u_k_criterion = nn.HuberLoss()
    u_i_criterion = nn.HuberLoss()


if __name__ == '__main__':
    main()
