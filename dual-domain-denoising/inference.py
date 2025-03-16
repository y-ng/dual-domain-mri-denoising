import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import fastmri
from fastmri.data import transforms
from constants import *
from models import UNet_kdata, UNet_image

np.random.seed(SEED)
torch.manual_seed(SEED)

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

    noisy_kdata_real = torch.tensor(noisy_kdata).real
    noisy_kdata_imag = torch.tensor(noisy_kdata).imag

    clean_kdata_real = torch.tensor(clean_kdata).real
    clean_kdata_imag = torch.tensor(clean_kdata).imag
    
    # load saved pytorch models for inference
    print('Loading models...')
    u_k_net_load = UNet_kdata()
    u_k_net_load.load_state_dict(torch.load(U_K_MODEL_PATH, weights_only=True))
    u_k_net_load.eval()
    with torch.no_grad():
        predicted_image = u_k_net_load(noisy_kdata)

    u_i_net_load = UNet_image()
    u_i_net_load.load_state_dict(torch.load(U_I_MODEL_PATH, weights_only=True))
    u_i_net_load.eval()

    # TODO: run inference on test data
    
    print('Done.')


if __name__ == '__main__':
    main()
