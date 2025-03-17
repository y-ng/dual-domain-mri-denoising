import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import fastmri
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
def show_coils(data, slice_nums, cmap=None):
    plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
    plt.tight_layout()
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
