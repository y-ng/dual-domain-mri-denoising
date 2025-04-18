import os
import numpy as np
import pickle as pk
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from constants import *
from models import UNet_kdata
from helpers import kspace_to_image, find_psnr, find_ssim

np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # load and format val data
    print('Loading test data...')
    with open(NOISY_KDATA_TEST, 'rb') as handle:
        noisy_kdata = pk.load(handle)
        print(noisy_kdata.shape)

    with open(CLEAN_KDATA_TEST, 'rb') as handle:
        clean_kdata = pk.load(handle)
        print(clean_kdata.shape)

    """
    with open(NOISY_KDATA_TEST2, 'rb') as handle:
        noisy_kdata = pk.load(handle)
        print(noisy_kdata.shape)

    with open(CLEAN_KDATA_TEST2, 'rb') as handle:
        clean_kdata = pk.load(handle)
        print(clean_kdata.shape)
    """

    noisy_kdata_real = torch.tensor(noisy_kdata).real
    noisy_kdata_imag = torch.tensor(noisy_kdata).imag

    clean_kdata_real = torch.tensor(clean_kdata).real
    clean_kdata_imag = torch.tensor(clean_kdata).imag

    dataset =  TensorDataset(
        torch.stack([noisy_kdata_real.to(torch.float32), noisy_kdata_imag.to(torch.float32)], dim=1), 
        torch.stack([clean_kdata_real.to(torch.float32), clean_kdata_imag.to(torch.float32)], dim=1)
    )
    dataloader = DataLoader(dataset)

    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    print(f'Device: {device}')
    
    # calculate metrics for KdM (mse)
    print('KdM (MSE):')
    u_k_net = UNet_kdata()
    u_k_net.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, 'u_k_net_mse_iter1_epoch200_batch4_lr1e-4.pt'), weights_only=True))
    u_k_net.to(device)
    u_k_net.eval()

    psnr, ssim = [], []
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)

        with torch.no_grad():
            u_k_outputs = u_k_net(noisy)

        noisy_complex = u_k_outputs[:, 0, :, :] + 1j * u_k_outputs[:, 1, :, :]
        noisy_complex = noisy_complex.to(CPU)

        noisy_alt = kspace_to_image(noisy_complex)
        noisy_alt[0, 128, 128] = torch.tensor(np.mean([
            noisy_alt[0, 128, 127],
            noisy_alt[0, 128, 129],
            noisy_alt[0, 127, 128],
            noisy_alt[0, 129, 128],
            noisy_alt[0, 127, 127],
            noisy_alt[0, 127, 129],
            noisy_alt[0, 129, 127],
            noisy_alt[0, 129, 129],
        ]))

        noisy_image = torch.reshape(noisy_alt, (noisy_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))
        clean_complex = clean[:, 0, :, :] + 1j * clean[:, 1, :, :]
        clean_complex = clean_complex.to(CPU)
        clean_image = torch.reshape(kspace_to_image(clean_complex), (clean_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))

        noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)

        psnr.append(find_psnr(noisy_image, clean_image))
        ssim.append(find_ssim(noisy_image, clean_image))

    print(f'Mean PSNR: {np.mean(psnr)}, {stats.norm.interval(0.95, loc=np.mean(psnr), scale=stats.sem(psnr))}')
    print(f'Mean SSIM: {np.mean(ssim)}, {stats.norm.interval(0.95, loc=np.mean(ssim), scale=stats.sem(ssim))}')

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_test_t1.pkl'), 'wb') as handle:
        pk.dump([psnr, ssim], handle, protocol=pk.HIGHEST_PROTOCOL)
    
    print('Done.')


if __name__ == '__main__':
    main()
