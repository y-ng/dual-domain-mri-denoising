import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from pathlib import Path
from constants import *

np.random.seed(SEED)

def main():
    # load arrays from pkl
    with open(MODEL_LOSS_PATH, 'rb') as handle:
        loss_data = pk.load(handle)

    u_k_epoch_loss = loss_data[0]
    u_k_step_loss = loss_data[1]
    u_k_val_loss = loss_data[2]
    u_i_epoch_loss = loss_data[3]
    u_i_step_loss = loss_data[4]
    u_i_val_loss = loss_data[5]
    u_i_val_ssim = loss_data[6]
    u_i_val_psnr = loss_data[7]

    # plot epoch loss
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axs[0].plot(u_k_epoch_loss, 'k-', label='Average Epoch Loss')
    axs[0].plot(u_k_val_loss, 'k--', label='Average Validation Loss')
    axs[0].set_ylabel('$U_k$ Net Loss')
    axs[0].set_ylim([0, 1e-5])
    axs[0].legend()
    axs[1].plot(u_i_epoch_loss, 'k-', label='Average Epoch Loss')
    axs[1].plot(u_i_val_loss, 'k--', label='Average Validation Loss')
    axs[1].set_xlabel('# of Epochs')
    axs[1].set_ylabel('$U_i$ Net Loss')
    axs[1].set_ylim([0, 1e-5])
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'epoch_loss.png'))
    plt.close()

    # plot step loss
    fig = plt.figure(figsize=(10, 6))
    plt.plot(u_k_step_loss, 'k-', label='U_k Net Step Loss')
    plt.plot(u_i_step_loss, 'k--', label='U_i Net Step Loss')
    plt.xlabel('# of Steps')
    plt.ylabel('Loss')
    plt.ylim([0, 1e-5])
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'step_loss.png'))
    plt.close()

    # plot ssim and psnr
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(u_i_val_ssim, 'k-', label='SSIM')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Structural Similiarity')
    ax2 = ax1.twinx()
    ax2.plot(u_i_val_psnr, 'k--', label='PSNR')
    ax2.set_ylabel('Peak Signal-to-Noise Ratio')
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_FOLDER, 'val_metrics.png'))
    plt.close()

    print('Done.')


if __name__ == '__main__':
    main()
