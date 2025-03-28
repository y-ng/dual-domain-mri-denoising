import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import fastmri
from fastmri.data import transforms
from pathlib import Path
from constants import *
from models import UNet_kdata, UNet_image
from helpers import crop_kspace, kspace_to_image, plot_noisy_vs_clean, show_coils, normalize_kspace

np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    # load saved pytorch models for inference
    print('Loading models...')
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    print(f'Device: {device}')

    u_k_net_load = UNet_kdata()
    u_k_net_load.load_state_dict(torch.load(U_K_MODEL_PATH, weights_only=True))
    u_k_net_load.to(device)
    u_k_net_load.eval()

    u_i_net_load = UNet_image()
    u_i_net_load.load_state_dict(torch.load(U_I_MODEL_PATH, weights_only=True))
    u_i_net_load.to(device)
    u_i_net_load.eval()

    # run inference on test data
    for i, file in enumerate(FILES_TEST):
        print(f'Running inference on {file} ({i + 1}/{len(FILES_TEST)})...')

        # load h5 file
        hf = h5py.File(os.path.join(DATA_FOLDER, file))
        volume_kspace = hf['kspace'][()]
        (n_slices, n_channels, height, width) = volume_kspace.shape
        volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)
        volume_kspace = normalize_kspace(volume_kspace)

        for n_slice in range(n_slices):
            print(f'Slice {n_slice + 1}/{n_slices}.')
            # add noise to kspace
            kspace = volume_kspace[n_slice]
            for n_channel in range(n_channels):
                noise_level = np.random.choice(NOISE_LEVELS)
                kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

            noisy_image_abs = kspace_to_image(kspace)

            # format complex values for model input
            kspace_real = torch.tensor(kspace).real
            kspace_imag = torch.tensor(kspace).imag
            u_k_net_in = torch.stack([kspace_real.to(torch.float32), kspace_imag.to(torch.float32)], dim=1)
            
            # run inference for kspace module
            u_k_net_in = u_k_net_in.to(device)
            with torch.no_grad():
                u_k_net_out = u_k_net_load(u_k_net_in)
            
            u_k_net_out = u_k_net_out.to(CPU)

            complex_output = u_k_net_out[:, 0, :, :] + 1j * u_k_net_out[:, 1, :, :]
            u_i_net_in = torch.reshape(kspace_to_image(complex_output), (n_channels, 1, CROP_SIZE, CROP_SIZE))

            """
            if n_slice == 0:
                show_coils(np.log(np.abs(kspace) + 1e-9), [0, 1, 2, 3])
                show_coils(np.log(np.abs(complex_output) + 1e-9), [0, 1, 2, 3])
                # show_coils(np.log(np.abs(complex_output - volume_kspace[n_slice]) + 1e-9), [0, 1, 2, 3])
                show_coils(kspace_to_image(complex_output), [0, 1, 2, 3], cmap='gray')
            """

            # run inference for image module
            u_i_net_in = u_i_net_in.to(device)
            with torch.no_grad():
                u_i_net_out = u_i_net_load(u_i_net_in)

            u_i_net_out = u_i_net_out.to(CPU)
            
            # combine multicoil data with root-sum-of-squares recon
            clean_image_rss = torch.reshape(fastmri.rss(u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
            # clean_image_rss[128, 128] = clean_image_rss[128, 127]
            noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)

            # save plot
            folder_path = os.path.join(OUTPUT_FOLDER, Path(file).stem)
            os.makedirs(folder_path, exist_ok=True)
            plot_noisy_vs_clean(noisy_image_rss, clean_image_rss, path=os.path.join(folder_path, f's{n_slice}.png'))

    
    print('Done.')


if __name__ == '__main__':
    main()
