import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch as torch
import fastmri
from constants import *
from models import UNet_kdata, UNet_image
from helpers import crop_kspace, kspace_to_image, normalize_kspace, show_coils

device = torch.device(CUDA if torch.cuda.is_available() else CPU)
print(f'Device: {device}')

# function to generate kdm output for the specified module from an appropriate input
def get_kdm_output(path_to_model, input):
    u_k_net = UNet_kdata()
    u_k_net.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, path_to_model), weights_only=True))
    u_k_net.to(device)
    u_k_net.eval()

    with torch.no_grad():
        input = input.to(device)
        u_k_net_out = u_k_net(input)

    u_k_net_out = u_k_net_out.to(CPU)
    
    complex_output = u_k_net_out[:, 0, :, :] + 1j * u_k_net_out[:, 1, :, :]
    image_output = kspace_to_image(complex_output)
    image_output[:, 128, 128] = torch.tensor(np.mean([
                image_output[0, 128, 127],
                image_output[0, 128, 129],
                image_output[0, 127, 128],
                image_output[0, 129, 128],
                image_output[0, 127, 127],
                image_output[0, 127, 129],
                image_output[0, 129, 127],
                image_output[0, 129, 129],
    ]))

    return torch.reshape(image_output, (input.shape[0], 1, CROP_SIZE, CROP_SIZE))


# function to generate ifm output for the specified module from an appropriate input
def get_idm_output(path_to_model, input):
    u_i_net = UNet_image()
    u_i_net.load_state_dict(torch.load(os.path.join(OUTPUT_FOLDER, path_to_model), weights_only=True))
    u_i_net.to(device)
    u_i_net.eval()

    with torch.no_grad():
        input = input.to(device)
        u_i_net_out = u_i_net(input)
    
    u_i_net_out = u_i_net_out.to(CPU)

    return u_i_net_out


def main():
    # plot for domain comparisons
    print('Running domain comparisons...')
    hf = h5py.File(os.path.join(DATA_FOLDER, FILES_VAL[0]))
    volume_kspace = hf['kspace'][()]
    (n_slices, n_channels, height, width) = volume_kspace.shape
    volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)
    volume_kspace = normalize_kspace(volume_kspace)

    clean_kspace = volume_kspace[2]
    noisy_kspace = clean_kspace.copy()
    for n_channel in range(n_channels):
        noise_level = np.random.choice(NOISE_LEVELS)
        noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

    clean_image_abs = kspace_to_image(clean_kspace)
    noisy_image_abs = kspace_to_image(noisy_kspace)

    kspace_real = torch.tensor(noisy_kspace).real
    kspace_imag = torch.tensor(noisy_kspace).imag
    u_k_net_in = torch.stack([kspace_real.to(torch.float32), kspace_imag.to(torch.float32)], dim=1)

    kdm_u_k_net_out = get_kdm_output('u_k_net_huber_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    kdm_u_i_net_out = kdm_u_k_net_out

    idm_u_k_net_out = u_k_net_in[:, 0, :, :] + 1j * u_k_net_in[:, 1, :, :]
    idm_u_k_net_out = kspace_to_image(idm_u_k_net_out)
    idm_u_k_net_out = torch.reshape(idm_u_k_net_out, (n_channels, 1, CROP_SIZE, CROP_SIZE))
    idm_u_i_net_out = get_idm_output('u_i_net_huber_iter1_epoch200_batch4_lr1e-4_img_only.pt', idm_u_k_net_out)
    
    wnet_u_k_net_out = get_kdm_output('u_k_net_huber_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    wnet_u_i_net_out = get_idm_output('u_i_net_huber_iter1_epoch200_batch4_lr1e-4.pt', wnet_u_k_net_out)

    kdm_image_rss = torch.reshape(fastmri.rss(kdm_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
    idm_image_rss = torch.reshape(fastmri.rss(idm_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
    wnet_image_rss = torch.reshape(fastmri.rss(wnet_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))

    clean_image_rss = fastmri.rss(clean_image_abs, dim=0)
    noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    box_x, box_y = 45, 90
    axs[0, 0].imshow(np.abs(noisy_image_rss.numpy()), cmap='gray')
    axs[0, 0].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 0].set_title('Noisy')
    axs[0, 0].set_ylabel('Full Image')
    axs[1, 0].imshow(np.abs(noisy_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[1, 0].set_ylabel('Zoomed In')
    axs[0, 1].imshow(np.abs(kdm_image_rss.numpy()), cmap='gray')
    axs[0, 1].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 1].set_title('KdM')
    axs[1, 1].imshow(np.abs(kdm_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 2].imshow(np.abs(idm_image_rss.numpy()), cmap='gray')
    axs[0, 2].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 2].set_title('IdM')
    axs[1, 2].imshow(np.abs(idm_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 3].imshow(np.abs(wnet_image_rss.numpy()), cmap='gray')
    axs[0, 3].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 3].set_title('WNet')
    axs[1, 3].imshow(np.abs(wnet_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 4].imshow(np.abs(clean_image_rss.numpy()), cmap='gray')
    axs[0, 4].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 4].set_title('Ground Truth')
    axs[1, 4].imshow(np.abs(clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    for i in range(axs.shape[0]):  
        for j in range(axs.shape[1]):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'comparison_domain.png'))
    plt.close()

    # plot for loss function comparisons
    print('Running loss function comparisons...')
    hf = h5py.File(os.path.join(DATA_FOLDER, FILES_VAL[-1]))
    volume_kspace = hf['kspace'][()]
    (n_slices, n_channels, height, width) = volume_kspace.shape
    volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)
    volume_kspace = normalize_kspace(volume_kspace)

    clean_kspace = volume_kspace[2]
    noisy_kspace = clean_kspace.copy()
    for n_channel in range(n_channels):
        noise_level = np.random.choice(NOISE_LEVELS)
        noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

    clean_image_abs = kspace_to_image(clean_kspace)
    noisy_image_abs = kspace_to_image(noisy_kspace)

    kspace_real = torch.tensor(noisy_kspace).real
    kspace_imag = torch.tensor(noisy_kspace).imag
    u_k_net_in = torch.stack([kspace_real.to(torch.float32), kspace_imag.to(torch.float32)], dim=1)

    kdm_huber_u_k_net_out = get_kdm_output('u_k_net_huber_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    kdm_huber_u_i_net_out = kdm_huber_u_k_net_out

    kdm_mse_u_k_net_out = get_kdm_output('u_k_net_mse_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    kdm_mse_u_i_net_out = kdm_mse_u_k_net_out
    
    wnet_huber_u_k_net_out = get_kdm_output('u_k_net_huber_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    wnet_huber_u_i_net_out = get_idm_output('u_i_net_huber_iter1_epoch200_batch4_lr1e-4.pt', wnet_huber_u_k_net_out)

    wnet_mse_u_k_net_out = get_kdm_output('u_k_net_mse_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    wnet_mse_u_i_net_out = get_idm_output('u_i_net_mse_iter1_epoch200_batch4_lr1e-4.pt', wnet_mse_u_k_net_out)

    kdm_huber_image_rss = torch.reshape(fastmri.rss(kdm_huber_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
    kdm_mse_image_rss = torch.reshape(fastmri.rss(kdm_mse_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
    wnet_huber_image_rss = torch.reshape(fastmri.rss(wnet_huber_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))
    wnet_mse_image_rss = torch.reshape(fastmri.rss(wnet_mse_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))

    clean_image_rss = fastmri.rss(clean_image_abs, dim=0)
    noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)

    fig, axs = plt.subplots(2, 6, figsize=(12, 4))
    box_x, box_y = 60, 30
    axs[0, 0].imshow(np.abs(noisy_image_rss.numpy()), cmap='gray')
    axs[0, 0].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 0].set_title('Noisy')
    axs[0, 0].set_ylabel('Full Image')
    axs[1, 0].imshow(np.abs(noisy_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[1, 0].set_ylabel('Zoomed In')
    axs[0, 1].imshow(np.abs(kdm_huber_image_rss.numpy()), cmap='gray')
    axs[0, 1].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 1].set_title('KdM (Huber)')
    axs[1, 1].imshow(np.abs(kdm_huber_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 2].imshow(np.abs(kdm_mse_image_rss.numpy()), cmap='gray')
    axs[0, 2].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 2].set_title('KdM (MSE)')
    axs[1, 2].imshow(np.abs(kdm_mse_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 3].imshow(np.abs(wnet_huber_image_rss.numpy()), cmap='gray')
    axs[0, 3].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 3].set_title('WNet (Huber)')
    axs[1, 3].imshow(np.abs(wnet_huber_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 4].imshow(np.abs(wnet_mse_image_rss.numpy()), cmap='gray')
    axs[0, 4].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 4].set_title('WNet (MSE)')
    axs[1, 4].imshow(np.abs(wnet_mse_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 5].imshow(np.abs(clean_image_rss.numpy()), cmap='gray')
    axs[0, 5].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 5].set_title('Ground Truth')
    axs[1, 5].imshow(np.abs(clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    for i in range(axs.shape[0]):  
        for j in range(axs.shape[1]):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'comparison_loss_func.png'))
    plt.close()

    # plot for test performance comparisons
    print('Running test performance comparisons...')
    hf = h5py.File(os.path.join(DATA_FOLDER, FILES_TEST[0]))
    volume_kspace = hf['kspace'][()]
    (n_slices, n_channels, height, width) = volume_kspace.shape
    volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)
    volume_kspace = normalize_kspace(volume_kspace)

    clean_kspace = volume_kspace[1]
    noisy_kspace = clean_kspace.copy()
    for n_channel in range(n_channels):
        noise_level = np.random.choice(NOISE_LEVELS)
        noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

    clean_image_abs = kspace_to_image(clean_kspace)
    noisy_image_abs = kspace_to_image(noisy_kspace)

    kspace_real = torch.tensor(noisy_kspace).real
    kspace_imag = torch.tensor(noisy_kspace).imag
    u_k_net_in = torch.stack([kspace_real.to(torch.float32), kspace_imag.to(torch.float32)], dim=1)

    kdm_mse_u_k_net_out = get_kdm_output('u_k_net_mse_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    kdm_mse_u_i_net_out = kdm_mse_u_k_net_out

    kdm_mse_image_rss = torch.reshape(fastmri.rss(kdm_mse_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))

    clean_image_rss = fastmri.rss(clean_image_abs, dim=0)
    noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)

    fig, axs = plt.subplots(3, 6, figsize=(12, 6))
    box_x, box_y = 175, 75
    axs[0, 0].imshow(np.abs(noisy_image_rss.numpy()), cmap='gray')
    axs[0, 0].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 0].set_title('Noisy (T1w)')
    axs[0, 0].set_ylabel('Full Image')
    axs[1, 0].imshow(np.abs(noisy_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[1, 0].set_ylabel('Zoomed In')
    axs[2, 0].imshow(np.abs(noisy_image_rss.numpy() - clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 0].set_ylabel('Residuals')
    axs[0, 1].imshow(np.abs(kdm_mse_image_rss.numpy()), cmap='gray')
    axs[0, 1].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 1].set_title('KdM (T1w)')
    axs[1, 1].imshow(np.abs(kdm_mse_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 1].imshow(np.abs(kdm_mse_image_rss.numpy() - clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 2].imshow(np.abs(clean_image_rss.numpy()), cmap='gray')
    axs[0, 2].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 2].set_title('Ground Truth (T1w)')
    axs[1, 2].imshow(np.abs(clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 2].set_axis_off()

    hf = h5py.File(os.path.join(DATA_FOLDER, FILES_TEST_T2[0]))
    volume_kspace = hf['kspace'][()]
    (n_slices, n_channels, height, width) = volume_kspace.shape
    volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)
    volume_kspace = normalize_kspace(volume_kspace)

    clean_kspace = volume_kspace[1]
    noisy_kspace = clean_kspace.copy()
    for n_channel in range(n_channels):
        noise_level = np.random.choice(NOISE_LEVELS)
        noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

    clean_image_abs = kspace_to_image(clean_kspace)
    noisy_image_abs = kspace_to_image(noisy_kspace)

    kspace_real = torch.tensor(noisy_kspace).real
    kspace_imag = torch.tensor(noisy_kspace).imag
    u_k_net_in = torch.stack([kspace_real.to(torch.float32), kspace_imag.to(torch.float32)], dim=1)

    kdm_mse_u_k_net_out = get_kdm_output('u_k_net_mse_iter1_epoch200_batch4_lr1e-4.pt', u_k_net_in)
    kdm_mse_u_i_net_out = kdm_mse_u_k_net_out

    kdm_mse_image_rss = torch.reshape(fastmri.rss(kdm_mse_u_i_net_out, dim=0), shape=(CROP_SIZE, CROP_SIZE))

    clean_image_rss = fastmri.rss(clean_image_abs, dim=0)
    noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)

    box_x, box_y = 45, 45
    axs[0, 3].imshow(np.abs(noisy_image_rss.numpy()), cmap='gray')
    axs[0, 3].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 3].set_title('Noisy (T2w)')
    axs[1, 3].imshow(np.abs(noisy_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 3].imshow(np.abs(noisy_image_rss.numpy() - clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 4].imshow(np.abs(kdm_mse_image_rss.numpy()), cmap='gray')
    axs[0, 4].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 4].set_title('KdM (T2w)')
    axs[1, 4].imshow(np.abs(kdm_mse_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 4].imshow(np.abs(kdm_mse_image_rss.numpy() - clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[0, 5].imshow(np.abs(clean_image_rss.numpy()), cmap='gray')
    axs[0, 5].add_patch(patches.Rectangle(
        (box_x, box_y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 5].set_title('Ground Truth (T2w)')
    axs[1, 5].imshow(np.abs(clean_image_rss.numpy())[box_y:(box_y + BOX_SIZE), box_x:(box_x + BOX_SIZE)], cmap='gray')
    axs[2, 5].set_axis_off()

    for i in range(axs.shape[0]):  
        for j in range(axs.shape[1]):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'comparison_test_performance.png'))
    plt.close()

    print('Done.')


if __name__ == '__main__':
    main()
