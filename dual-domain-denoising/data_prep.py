import os
import h5py
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms
from constants import *
from helpers import crop_kspace, show_coils, kspace_to_image

np.random.seed(SEED)

def main():
    
    noisy_kdata, clean_kdata = None, None
    noisy_image, clean_image = None, None
    
    for i, file in enumerate(FILES_TRAIN):
        print(f'Processing file {i + 1} / {len(FILES_TRAIN)}...')
        # load h5 file
        hf = h5py.File(os.path.join(DATA_FOLDER, file))

        # select kspace key 
        volume_kspace = hf['kspace'][()]
        # shape should be num slices, num channels, height, width
        (n_slices, n_channels, height, width) = volume_kspace.shape
        print(n_slices, n_channels)

        volume_kspace = crop_kspace(volume_kspace, CROP_SIZE)

        # assign empty arrays if first file
        if noisy_kdata is None:
            noisy_kdata = np.empty(shape=(0, CROP_SIZE, CROP_SIZE))
            clean_kdata = np.empty(shape=(0, CROP_SIZE, CROP_SIZE))
            noisy_image = np.empty(shape=(0, CROP_SIZE, CROP_SIZE))
            clean_image = np.empty(shape=(0, CROP_SIZE, CROP_SIZE))

        # select each slice in a volume
        for n_slice in range(n_slices):
            clean_kspace = volume_kspace[n_slice]

            # add noise to kspace channels
            noisy_kspace = clean_kspace.copy()
            for n_channel in range(n_channels):
                noise_level = np.random.choice(NOISE_LEVELS)
                noisy_kspace[n_channel] += np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE)) + 1j * np.random.normal(0, noise_level, size=(CROP_SIZE, CROP_SIZE))

            # track kspace data for training
            noisy_kdata = np.concatenate((noisy_kdata, noisy_kspace), axis=0)
            clean_kdata = np.concatenate((clean_kdata, clean_kspace), axis=0)

            """
            show_coils(np.log(np.abs(clean_kspace) + 1e-9), [0, 1, 2, 3])
            show_coils(np.log(np.abs(noisy_kspace) + 1e-9), [0, 1, 2, 3])
            """

            # absolute value of img from kspace 
            clean_image_abs = kspace_to_image(clean_kspace)
            noisy_image_abs = kspace_to_image(noisy_kspace)

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
            axs[0].imshow(np.abs(clean_image_rss.numpy()), cmap='gray')
            axs[1].imshow(np.abs(noisy_image_rss.numpy()), cmap='gray')
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
