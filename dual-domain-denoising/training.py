import numpy as np
import pickle as pk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import fastmri
from fastmri.data import transforms
from constants import *
from models import UNet_kdata, UNet_image
from helpers import kspace_to_image, plot_noisy_vs_clean, find_ssim, find_psnr

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

    with open(NOISY_KDATA_VAL, 'rb') as handle:
        noisy_kdata_val = pk.load(handle)
        print(noisy_kdata_val.dtype, noisy_kdata_val.shape)

    with open(CLEAN_KDATA_VAL, 'rb') as handle:
        clean_kdata_val = pk.load(handle)
        print(clean_kdata_val.dtype, clean_kdata_val.shape)
        

    (n_samples, height, width) = noisy_kdata.shape
    print(f'Number of training samples: {n_samples}')
    print(f'Image size: {height}x{width}')

    (n_samples_val, height_val, width_val) = noisy_kdata_val.shape
    print(f'Number of validation samples: {n_samples_val}')
    print(f'Image size: {height_val}x{width_val}')

    # set up model
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    print(f'Device: {device}')

    u_k_net = UNet_kdata()
    u_i_net = UNet_image()

    u_k_net.to(device)
    u_i_net.to(device)

    batch_size = 4

    # format train data
    noisy_kdata_real = torch.tensor(noisy_kdata).real
    noisy_kdata_imag = torch.tensor(noisy_kdata).imag

    clean_kdata_real = torch.tensor(clean_kdata).real
    clean_kdata_imag = torch.tensor(clean_kdata).imag

    kdata_train_data = TensorDataset(
        torch.stack([noisy_kdata_real.to(torch.float32), noisy_kdata_imag.to(torch.float32)], dim=1), 
        torch.stack([clean_kdata_real.to(torch.float32), clean_kdata_imag.to(torch.float32)], dim=1)
    )
    kdata_train_loader = DataLoader(kdata_train_data, batch_size=batch_size, shuffle=True)

    # format val data
    noisy_val_real = torch.tensor(noisy_kdata_val).real
    noisy_val_imag = torch.tensor(noisy_kdata_val).imag

    clean_val_real = torch.tensor(clean_kdata_val).real
    clean_val_imag = torch.tensor(clean_kdata_val).imag
    
    kdata_val_data = TensorDataset(
        torch.stack([noisy_val_real.to(torch.float32), noisy_val_imag.to(torch.float32)], dim=1), 
        torch.stack([clean_val_real.to(torch.float32), clean_val_imag.to(torch.float32)], dim=1)
    )
    kdata_val_loader = DataLoader(kdata_val_data, batch_size=1)
    
    epochs_per_iter = 50
    num_iters = 1 # num_epochs = num_iters * epochs_per_iter (* 2 models)

    u_k_optimizer = torch.optim.Adam(u_k_net.parameters(), lr=1e-4)
    u_i_optimizer = torch.optim.Adam(u_i_net.parameters(), lr=1e-4)
    u_k_criterion = nn.HuberLoss()
    u_i_criterion = nn.HuberLoss()

    u_k_epoch_loss, u_k_step_loss = [], []
    u_k_val_loss = []
    u_i_epoch_loss, u_i_step_loss = [], []
    u_i_val_loss, u_i_val_ssim, u_i_val_psnr = [], [], []

    # train model (n epochs for u_k + n epochs for u_i / iter)
    for iter in range(num_iters):
        print(f'Iteration {iter + 1}/{num_iters}')
        
        # training for u_k_net
        for i in range(epochs_per_iter):
            print(f'Epoch {iter * epochs_per_iter + i + 1}/{num_iters * epochs_per_iter} for U_k')
            u_k_net.train()

            epoch_loss = 0
            step = 0

            for noisy_kspace, clean_kspace in kdata_train_loader:
                step += 1

                """
                # check if inputs look correct
                noisy_test = noisy_kspace[0, 0, :, :] + 1j * noisy_kspace[0, 1, :, :]
                clean_test = clean_kspace[0, 0, :, :] + 1j * clean_kspace[0, 1, :, :]
                plot_noisy_vs_clean(np.log(np.abs(noisy_test) + 1e-9), np.log(np.abs(clean_test) + 1e-9))
                plot_noisy_vs_clean(kspace_to_image(noisy_test), kspace_to_image(clean_test))
                """

                noisy_kspace, clean_kspace = noisy_kspace.to(device), clean_kspace.to(device)
                u_k_net_outputs = u_k_net(noisy_kspace)
                u_k_net_loss = u_k_criterion(u_k_net_outputs, clean_kspace)

                """
                # check u_k_net output
                noisy_test = u_k_net_outputs.to(CPU)[0, 0, :, :] + 1j * u_k_net_outputs.to(CPU)[0, 1, :, :]
                clean_test = clean_kspace.to(CPU)[0, 0, :, :] + 1j * clean_kspace.to(CPU)[0, 1, :, :]
                plot_noisy_vs_clean(np.log(np.abs(noisy_test.detach()) + 1e-9), np.log(np.abs(clean_test.detach()) + 1e-9))
                plot_noisy_vs_clean(kspace_to_image(noisy_test.detach()), kspace_to_image(clean_test.detach()))
                """

                # update weights
                u_k_optimizer.zero_grad()
                u_k_net_loss.backward()
                u_k_optimizer.step()

                epoch_loss += u_k_net_loss.item()
                u_k_step_loss.append(u_k_net_loss.item())
            
            epoch_loss /= step
            print(f'----- train loss: {epoch_loss}')
            u_k_epoch_loss.append(epoch_loss)

            # run validation
            u_k_net.eval()
            val_loss = 0
            with torch.no_grad():
                for noisy_val, clean_val in kdata_val_loader:
                    noisy_val, clean_val = noisy_val.to(device), clean_val.to(device)
                    val_outputs = u_k_net(noisy_val)
                    val_loss += u_k_criterion(val_outputs, clean_val).item()

            val_loss /= n_samples_val
            print(f'----- val loss: {val_loss}')
            u_k_val_loss.append(val_loss)
        
        
        # training for u_i_net
        for i in range(epochs_per_iter):
            print(f'Epoch {iter * epochs_per_iter + i + 1}/{num_iters * epochs_per_iter} for U_i')
            u_i_net.train()

            epoch_loss = 0
            step = 0

            for noisy_kspace, clean_kspace in kdata_train_loader:
                step += 1

                noisy_kspace, clean_kspace = noisy_kspace.to(device), clean_kspace.to(device)
                
                u_k_net.eval()
                with torch.no_grad():
                    u_k_net_outputs = u_k_net(noisy_kspace)

                # kspace to image conversion
                noisy_complex = u_k_net_outputs[:, 0, :, :] + 1j * u_k_net_outputs[:, 1, :, :]
                noisy_complex = noisy_complex.to(CPU)
                noisy_image = torch.reshape(kspace_to_image(noisy_complex), (noisy_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))
                clean_complex = clean_kspace[:, 0, :, :] + 1j * clean_kspace[:, 1, :, :]
                clean_complex = clean_complex.to(CPU)
                clean_image = torch.reshape(kspace_to_image(clean_complex), (clean_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))

                """
                # check input to u_i_net
                plot_noisy_vs_clean(noisy_image[0, 0].detach(), clean_image[0, 0].detach())
                """

                noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
                
                u_i_net_outputs = u_i_net(noisy_image)
                u_i_net_loss = u_i_criterion(u_i_net_outputs, clean_image)

                """
                # check output of u_i_net
                plot_noisy_vs_clean(u_i_net_outputs.to(CPU)[0, 0].detach(), clean_image.to(CPU)[0, 0].detach())
                """

                # update weights
                u_i_optimizer.zero_grad()
                u_i_net_loss.backward()
                u_i_optimizer.step()

                epoch_loss += u_i_net_loss.item()
                u_i_step_loss.append(u_i_net_loss.item())

            epoch_loss /= step
            print(f'----- train loss: {epoch_loss}')
            u_i_epoch_loss.append(epoch_loss)

            # run validation
            u_i_net.eval()
            val_loss, val_ssim, val_psnr = 0, 0, 0
            with torch.no_grad():
                for noisy_val, clean_val in kdata_val_loader:
                    noisy_val, clean_val = noisy_val.to(device), clean_val.to(device)
                    u_k_net_outputs = u_k_net(noisy_val)

                    val_noisy_complex = u_k_net_outputs[:, 0, :, :] + 1j * u_k_net_outputs[:, 1, :, :]
                    val_noisy_complex = val_noisy_complex.to(CPU)
                    val_noisy_image = torch.reshape(kspace_to_image(val_noisy_complex), (val_noisy_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))
                    val_clean_complex = clean_val[:, 0, :, :] + 1j * clean_val[:, 1, :, :]
                    val_clean_complex = val_clean_complex.to(CPU)
                    val_clean_image = torch.reshape(kspace_to_image(val_clean_complex), (val_clean_complex.shape[0], 1, CROP_SIZE, CROP_SIZE))

                    val_noisy_image, val_clean_image = val_noisy_image.to(device), val_clean_image.to(device)
                    
                    val_outputs = u_i_net(val_noisy_image)
                    val_loss += u_k_criterion(val_outputs, val_clean_image).item()
                    val_ssim += find_ssim(val_outputs, val_clean_image)
                    val_psnr += find_psnr(val_outputs, val_clean_image)

            val_loss /= n_samples_val
            val_ssim /= n_samples_val
            val_psnr /= n_samples_val
            print(f'----- val loss: {val_loss}, ssim: {val_ssim}, psnr: {val_psnr}')
            u_i_val_loss.append(val_loss)
            u_i_val_ssim.append(val_ssim)
            u_i_val_psnr.append(val_psnr)


    # save final models
    print('Saving models and training outputs...')
    u_k_net.to('cpu')
    u_i_net.to('cpu')
    torch.save(u_k_net.state_dict(), U_K_MODEL_PATH)
    torch.save(u_i_net.state_dict(), U_I_MODEL_PATH)
    with open(MODEL_LOSS_PATH, 'wb') as handle:
        pk.dump(
            [
                u_k_epoch_loss, u_k_step_loss, u_k_val_loss, 
                u_i_epoch_loss, u_i_step_loss, u_i_val_loss, 
                u_i_val_ssim, u_i_val_psnr
            ], 
            handle, 
            protocol=pk.HIGHEST_PROTOCOL
        )
    
    print('Done.')


if __name__ == '__main__':
    main()
