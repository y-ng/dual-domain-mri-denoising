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

    # set up model
    u_k_net = UNet_kdata()
    u_i_net = UNet_image()

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

    num_iters = 1 # num_batches = num_iters * 5

    u_k_optimizer = torch.optim.Adam(u_k_net.parameters(), lr=1e-4)
    u_i_optimizer = torch.optim.Adam(u_i_net.parameters(), lr=1e-4)
    u_k_criterion = nn.HuberLoss()
    u_i_criterion = nn.HuberLoss()

    # train models
    # TODO
    for iter in range(num_iters):
        print(f'Iteration {iter + 1}/{num_iters}')
        
        for i in range(5):
            print(f'Epoch {(i + 1) * (iter + 1)}/{num_iters * 5} for U_k')
            for noisy, clean in kdata_train_loader:
                u_k_net_outputs = u_k_net(noisy)
                u_k_net_loss = u_k_criterion(u_k_net_outputs, clean)

                # update weights
                u_k_optimizer.zero_grad()
                u_k_net_loss.backward()
                u_k_optimizer.step()


    # save final models
    print('Saving models and training outputs...')
    torch.save(u_k_net.state_dict(), U_K_MODEL_PATH)
    torch.save(u_i_net.state_dict(), U_I_MODEL_PATH)
    

    """
    # load saved pytorch models for inference
    print('Loading models...')
    u_k_net_load = UNet_kdata()
    u_k_net_load.load_state_dict(torch.load(U_K_MODEL_PATH, weights_only=True))
    u_k_net_load.eval()

    u_i_net_load = UNet_image()
    u_i_net_load.load_state_dict(torch.load(U_I_MODEL_PATH, weights_only=True))
    u_i_net_load.eval()

    print('Done.')"
    """


if __name__ == '__main__':
    main()
