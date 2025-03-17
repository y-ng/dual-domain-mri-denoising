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
from helpers import kspace_to_image

np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # load pkl data
    print('Read in data from files...')
    with open(NOISY_KDATA_PATH, 'rb') as handle:
        noisy_kdata = pk.load(handle)
        print(noisy_kdata.dtype, noisy_kdata.shape)

    noisy_kdata_real = torch.tensor(noisy_kdata).real
    noisy_kdata_imag = torch.tensor(noisy_kdata).imag

    kdata_test_data = torch.stack([noisy_kdata_real.to(torch.float32), noisy_kdata_imag.to(torch.float32)], dim=1)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # load saved pytorch models for inference
    print('Loading models...')
    u_k_net_load = UNet_kdata()
    u_k_net_load.load_state_dict(torch.load(U_K_MODEL_PATH, weights_only=True))
    u_k_net_load.to(device)
    u_k_net_load.eval()
    with torch.no_grad():
        predicted_image = u_k_net_load(noisy_kdata)

    u_i_net_load = UNet_image()
    u_i_net_load.load_state_dict(torch.load(U_I_MODEL_PATH, weights_only=True))
    u_i_net_load.to(device)
    u_i_net_load.eval()

    # TODO: run inference on test data
    
    print('Done.')


if __name__ == '__main__':
    main()
