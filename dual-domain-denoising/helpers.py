import numpy as np
import matplotlib.pyplot as plt
import fastmri
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


# function to plot absolute value of kspace
def show_coils(data, slice_nums, cmap=None, path=None):
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
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
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.abs(clean_image.numpy()), cmap='gray')
    axs[1].imshow(np.abs(noisy_image.numpy()), cmap='gray')
    plt.tight_layout()

    if path:
        fig.savefig(path)
        plt.close()
    else:
        plt.show()


# function to calculate structural similarity b/w two images
def find_ssim(image, prediction):
    metric = StructuralSimilarityIndexMeasure()
    return metric(prediction, image)


# function to calculate peak signal-to-noise ratio of an image
def find_psnr(image, prediction):
    metric = PeakSignalNoiseRatio()
    return metric(prediction, image)

