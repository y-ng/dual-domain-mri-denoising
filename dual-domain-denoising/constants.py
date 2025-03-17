# path for training and output data
TRAIN_FOLDER = './dual-domain-denoising/training_data'
OUTPUT_FOLDER = './dual-domain-denoising/outputs'

# paths for noisy/clean kspace/image data
NOISY_KDATA_PATH = TRAIN_FOLDER + '/noisy_train_kdata.pkl'
CLEAN_KDATA_PATH = TRAIN_FOLDER + '/clean_train_kdata.pkl'
NOISY_IMAGE_PATH = TRAIN_FOLDER + '/noisy_train_image.pkl'
CLEAN_IMAGE_PATH = TRAIN_FOLDER + '/clean_train_image.pkl'

# path for model outputs
U_K_MODEL_PATH = OUTPUT_FOLDER + '/u_k_net.pt'
U_I_MODEL_PATH = OUTPUT_FOLDER + '/u_i_net.pt'
MODEL_LOSS_PATH = OUTPUT_FOLDER + '/model_loss.pkl'

# seed for random state
SEED = 42

# noise levels for additive kspace noise
NOISE_LEVELS = [5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5]

# input image size
CROP_SIZE = 256
