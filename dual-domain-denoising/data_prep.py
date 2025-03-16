import h5py
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms
from constants import *

np.random.seed(SEED)

# function to plot absolute value of kspace
def show_coils(data, slice_nums, cmap=None):
    plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=cmap)
    plt.tight_layout()
    plt.show()


def main():
    
    noisy_kdata, clean_kdata = None, None
    noisy_image, clean_image = None, None
    
    files = [
        './file_brain_AXT2_202_2020179.h5',
        './file_brain_AXT2_207_2070513.h5',
    ]
    
    for i, file in enumerate(files):
        print(f'Processing file {i + 1} / {len(files)}...')
        # load h5 file
        hf = h5py.File(file)

        # select kspace key 
        volume_kspace = hf['kspace'][()]
        # shape should be num slices, num channels, height, width
        (n_slices, n_channels, height, width) = volume_kspace.shape
        print(n_slices, n_channels)

        # sample every other k-space line in x-direction to go from rectangular to square k-space
        # then crop centre to reduce resolution to 256x256
        volume_kspace = volume_kspace[:, :, 1::2, :]
        volume_kspace = transforms.center_crop(volume_kspace, shape=(256, 256))

        # assign empty arrays if first file
        if noisy_kdata is None:
            noisy_kdata = np.empty(shape=(0, height, width))
            clean_kdata = np.empty(shape=(0, height, width))
            noisy_image = np.empty(shape=(0, height, width))
            clean_image = np.empty(shape=(0, height, width))

        # select each slice in a volume
        for n_slice in range(n_slices):
            clean_kspace = volume_kspace[n_slice]

            # add noise to kspace channels
            noisy_kspace = clean_kspace.copy()
            for n_channel in range(n_channels):
                noise_level = np.random.choice(NOISE_LEVELS)
                noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(height, width)) + 1j * np.random.normal(0, noise_level, size=(height, width))

            # track kspace data for training
            noisy_kdata = np.concatenate((noisy_kdata, noisy_kspace), axis=0)
            clean_kdata = np.concatenate((clean_kdata, clean_kspace), axis=0)

            """
            show_coils(np.log(np.abs(clean_kspace) + 1e-9), [0, 1, 2, 3])
            show_coils(np.log(np.abs(noisy_kspace) + 1e-9), [0, 1, 2, 3])
            """
            
            # np array to pytorch tensor
            clean_kspace_tensor = transforms.to_tensor(clean_kspace)
            noisy_kspace_tensor = transforms.to_tensor(noisy_kspace)

            # inverse fourier to get complex img 
            clean_image_data = fastmri.ifft2c(clean_kspace_tensor)
            noisy_image_data = fastmri.ifft2c(noisy_kspace_tensor)

            # absolute value of img
            clean_image_abs = fastmri.complex_abs(clean_image_data)
            noisy_image_abs = fastmri.complex_abs(noisy_image_data)

            """
            show_coils(clean_image_abs, [0, 1, 2, 3], cmap='gray')
            show_coils(noisy_image_abs, [0, 1, 2, 3], cmap='gray')
            """

            # track image data for training
            clean_image = np.concatenate((clean_image, clean_image_abs), axis=0)
            noisy_image = np.concatenate((noisy_image, noisy_image_abs), axis=0)
            
            """
            # combine multicoil data with root-sum-of-squares recon
            clean_image_rss = fastmri.rss(clean_image_abs, dim=0)
            noisy_image_rss = fastmri.rss(noisy_image_abs, dim=0)
            
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(np.abs(clean_image_rss.numpy())[250:350, 75:175], cmap='gray')
            axs[1].imshow(np.abs(noisy_image_rss.numpy())[250:350, 75:175], cmap='gray')
            plt.tight_layout()
            plt.show()
            """

    # writing all training data to pkl 
    print('Writing data to files...')
    with open(NOISY_KDATA_PATH, 'wb') as handle:
        pk.dump(noisy_kdata, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open(CLEAN_KDATA_PATH, 'wb') as handle:
        pk.dump(clean_kdata, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open(NOISY_IMAGE_PATH, 'wb') as handle:
        pk.dump(noisy_image, handle, protocol=pk.HIGHEST_PROTOCOL)

    with open(CLEAN_IMAGE_PATH, 'wb') as handle:
        pk.dump(clean_image, handle, protocol=pk.HIGHEST_PROTOCOL)
    
    print('Done.')


if __name__ == '__main__':
    main()
