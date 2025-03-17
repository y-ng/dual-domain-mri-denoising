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

    with open(CLEAN_KDATA_PATH, 'rb') as handle:
        clean_kdata = pk.load(handle)
        print(clean_kdata.dtype, clean_kdata.shape)

    """
    with open(NOISY_IMAGE_PATH, 'rb') as handle:
        noisy_image = pk.load(handle)
        print(noisy_image.dtype, noisy_image.shape)

    with open(CLEAN_IMAGE_PATH, 'rb') as handle:
        clean_image = pk.load(handle)
        print(clean_image.dtype, clean_image.shape)
    """

    (n_samples, height, width) = noisy_kdata.shape
    print(f'Number of training samples: {n_samples}')
    print(f'Image size: {height}x{width}')

    # set up model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    u_k_net = UNet_kdata()
    u_i_net = UNet_image()

    u_k_net.to(device)
    u_i_net.to(device)

    batch_size = 4

    noisy_kdata_real = torch.tensor(noisy_kdata).real
    noisy_kdata_imag = torch.tensor(noisy_kdata).imag

    clean_kdata_real = torch.tensor(clean_kdata).real
    clean_kdata_imag = torch.tensor(clean_kdata).imag

    kdata_train_data = TensorDataset(
        torch.stack([noisy_kdata_real.to(torch.float32), noisy_kdata_imag.to(torch.float32)], dim=1), 
        torch.stack([clean_kdata_real.to(torch.float32), clean_kdata_imag.to(torch.float32)], dim=1)
    )
    kdata_train_loader = DataLoader(kdata_train_data, batch_size=batch_size, shuffle=True)

    num_iters = 1 # num_epochs = num_iters * 5 * 2

    u_k_optimizer = torch.optim.Adam(u_k_net.parameters(), lr=1e-4)
    u_i_optimizer = torch.optim.Adam(u_i_net.parameters(), lr=1e-4)
    u_k_criterion = nn.HuberLoss()
    u_i_criterion = nn.HuberLoss()

    # TODO: train model
    for iter in range(num_iters):
        print(f'Iteration {iter + 1}/{num_iters}')
        
        for i in range(5):
            print(f'Epoch {(i + 1) * (iter + 1)}/{num_iters * 5} for U_k')
            for data in kdata_train_loader:
                noisy, clean = data[0].to(device), data[1].to(device)
                u_k_net_outputs = u_k_net(noisy)
                u_k_net_loss = u_k_criterion(u_k_net_outputs, clean)

                # update weights
                u_k_optimizer.zero_grad()
                u_k_net_loss.backward()
                u_k_optimizer.step()


    # save final models
    print('Saving models and training outputs...')
    u_k_net.to('cpu')
    u_i_net.to('cpu')
    torch.save(u_k_net.state_dict(), U_K_MODEL_PATH)
    torch.save(u_i_net.state_dict(), U_I_MODEL_PATH)
    
    print('Done.')


if __name__ == '__main__':
    main()
