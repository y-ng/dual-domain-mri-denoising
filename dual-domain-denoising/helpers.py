import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fastmri
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fastmri.data import transforms
from constants import *

np.random.seed(SEED)

# function to crop k-space for lower resolution image
def crop_kspace(volume_kspace, size):
    # sample every other k-space line in x-direction to go from rectangular to square k-space
    volume_kspace = volume_kspace[:, :, 1::2, :]
    # crop centre to reduce resolution to size x size
    volume_kspace = transforms.center_crop(volume_kspace, shape=(size, size))

    return volume_kspace


# function to normalize k-space
def normalize_kspace(kspace):
    """
    real = np.real(kspace)
    imag = np.imag(kspace)

    real = (real - np.mean(real)) / np.std(real)
    imag = (imag - np.mean(imag)) / np.std(imag)

    return real + 1j * imag
    """

    abs = np.abs(kspace)
    norm = abs / np.max(abs)
    rads = np.angle(kspace)

    return (norm * np.cos(rads)) + 1j * (norm * np.sin(rads))


# function to plot absolute value of kspace
def show_coils(data, coil_nums, cmap=None, path=None):
    fig = plt.figure()
    for i, num in enumerate(coil_nums):
        plt.subplot(1, len(coil_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'C{num}')
    plt.tight_layout()

    if path:
        fig.savefig(path)
        plt.close()
    else:
        plt.show()


# function to convert kspace to image space
def kspace_to_image(kdata):
    # np array to pytorch tensor
    kspace_tensor = transforms.to_tensor(kdata)
    # inverse fourier to get complex img 
    image_data = fastmri.ifft2c(kspace_tensor)
    # absolute value of img
    image_abs = fastmri.complex_abs(image_data)

    return image_abs


# function to compare clean vs noisy reconstructed images
def plot_noisy_vs_clean(noisy_image, clean_image, path=None):
    fig, axs = plt.subplots(2, 2)

    # plot full image
    axs[0, 0].imshow(np.abs(noisy_image.numpy()), cmap='gray')
    axs[0, 0].add_patch(patches.Rectangle(
        (BOX_X, BOX_Y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 1].imshow(np.abs(clean_image.numpy()), cmap='gray')
    axs[0, 1].add_patch(patches.Rectangle(
        (BOX_X, BOX_Y), BOX_SIZE, BOX_SIZE, linewidth=1, edgecolor='r', facecolor='none'
    ))
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # plot zoomed image
    axs[1, 0].imshow(np.abs(noisy_image.numpy())[BOX_Y:(BOX_Y + BOX_SIZE), BOX_X:(BOX_X + BOX_SIZE)], cmap='gray')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 1].imshow(np.abs(clean_image.numpy())[BOX_Y:(BOX_Y + BOX_SIZE), BOX_X:(BOX_X + BOX_SIZE)], cmap='gray')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # TODO: add scale to image (e.g., how much is a mm) https://pypi.org/project/matplotlib-scalebar/ 
    # ITK snap for visualization --> images currently too dark
    plt.tight_layout()

    if path:
        fig.savefig(path)
        plt.close()
    else:
        plt.show()


# function to calculate structural similarity b/w two images
def find_ssim(predicted, target):
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    metric = StructuralSimilarityIndexMeasure().to(device)
    ssim = metric(target, predicted).to(CPU)
    metric.to(CPU)
    return ssim


# function to calculate peak signal-to-noise ratio of an image
def find_psnr(predicted, target):
    device = torch.device(CUDA if torch.cuda.is_available() else CPU)
    metric = PeakSignalNoiseRatio().to(device)
    psnr = metric(target, predicted).to(CPU)
    metric.to(CPU)
    return psnr

