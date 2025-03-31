import os
import numpy as np 

CPU = 'cpu'
CUDA = 'cuda:0'

# path for input and output data
TRAIN_FOLDER = './dual-domain-denoising/training_data'
OUTPUT_FOLDER = './dual-domain-denoising/outputs'
DATA_FOLDER = './data'

# paths for noisy/clean kspace data
NOISY_KDATA_PATH = os.path.join(TRAIN_FOLDER, 'noisy_train_kdata.pkl')
CLEAN_KDATA_PATH = os.path.join(TRAIN_FOLDER, 'clean_train_kdata.pkl')

NOISY_KDATA_VAL = os.path.join(TRAIN_FOLDER, 'noisy_val_kdata.pkl')
CLEAN_KDATA_VAL = os.path.join(TRAIN_FOLDER, 'clean_val_kdata.pkl')

NOISY_KDATA_TEST = os.path.join(TRAIN_FOLDER, 'noisy_test_kdata.pkl')
CLEAN_KDATA_TEST = os.path.join(TRAIN_FOLDER, 'clean_test_kdata.pkl')

NOISY_KDATA_TEST2 = os.path.join(TRAIN_FOLDER, 'noisy_test2_kdata.pkl')
CLEAN_KDATA_TEST2 = os.path.join(TRAIN_FOLDER, 'clean_test2_kdata.pkl')

# path for model outputs
U_K_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'u_k_net.pt')
U_I_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'u_i_net.pt')
MODEL_LOSS_PATH = os.path.join(OUTPUT_FOLDER, 'model_loss.pkl')

# seed for random state
SEED = 42

# noise levels for additive kspace noise
NOISE_LEVELS = np.linspace(1e-3, 5e-3, 9)

# input image size
CROP_SIZE = 256

# for plotting
BOX_SIZE = 64
BOX_X, BOX_Y = 150, 50

# files names for train/val/test
FILES_TRAIN = [
    'file_brain_AXT1_201_6002717.h5',
    'file_brain_AXT1_201_6002779.h5',
    'file_brain_AXT1_201_6002836.h5',
    'file_brain_AXT1_202_2020098.h5',
    'file_brain_AXT1_202_2020109.h5',
    'file_brain_AXT1_202_2020131.h5',
    'file_brain_AXT1_202_2020146.h5',
    'file_brain_AXT1_202_2020186.h5',
    'file_brain_AXT1_202_2020209.h5',
    'file_brain_AXT1_202_2020233.h5',
    'file_brain_AXT1_202_2020334.h5',
    'file_brain_AXT1_202_2020377.h5',
    'file_brain_AXT1_202_2020384.h5',
    'file_brain_AXT1_202_2020389.h5',
    'file_brain_AXT1_202_2020391.h5',
]

FILES_VAL = [
    'file_brain_AXT1_202_2020478.h5',
    'file_brain_AXT1_202_2020486.h5',
    'file_brain_AXT1_202_2020496.h5',
    'file_brain_AXT1_202_2020509.h5',
    'file_brain_AXT1_202_2020570.h5',
]

FILES_TEST = [
    'file_brain_AXT1_202_6000305.h5',
    'file_brain_AXT1_202_6000312.h5',
    'file_brain_AXT1_202_6000339.h5',
    'file_brain_AXT1_202_6000347.h5',
    'file_brain_AXT1_202_6000382.h5',
]

FILES_TEST_T2 = [
    'file_brain_AXT2_200_200020.h5',
    'file_brain_AXT2_200_200057.h5',
    'file_brain_AXT2_200_200080.h5',
    'file_brain_AXT2_200_200092.h5',
    'file_brain_AXT2_200_200175.h5',
]
