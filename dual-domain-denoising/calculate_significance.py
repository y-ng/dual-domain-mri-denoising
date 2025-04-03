import os
import numpy as np
import pickle as pk
import scipy.stats as stats
from constants import *

def main():
    with open(os.path.join(OUTPUT_FOLDER, 'metrics_noisy.pkl'), 'rb') as handle:
        psnr_noisy, ssim_noisy = pk.load(handle)
    
    with open(os.path.join(OUTPUT_FOLDER, 'metrics_kdm.pkl'), 'rb') as handle:
        psnr_kdm, ssim_kdm = pk.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_idm.pkl'), 'rb') as handle:
        psnr_idm, ssim_idm = pk.load(handle)
    
    with open(os.path.join(OUTPUT_FOLDER, 'metrics_wnet.pkl'), 'rb') as handle:
        psnr_wnet, ssim_wnet = pk.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_kdm_mse.pkl'), 'rb') as handle:
        psnr_kdm_mse, ssim_kdm_mse = pk.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_wnet_mse.pkl'), 'rb') as handle:
        psnr_wnet_mse, ssim_wnet_mse = pk.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_test_t1.pkl'), 'rb') as handle:
        psnr_test_t1, ssim_test_t1 = pk.load(handle)

    with open(os.path.join(OUTPUT_FOLDER, 'metrics_test_t2.pkl'), 'rb') as handle:
        psnr_test_t2, ssim_test_t2 = pk.load(handle)

    # check for normality
    print('Checking for normal dist...')
    print(f'psnr_noisy {stats.shapiro(psnr_noisy)}')
    print(f'ssim_noisy {stats.shapiro(ssim_noisy)}')
    print(f'psnr_kdm {stats.shapiro(psnr_kdm)}')
    print(f'ssim_kdm {stats.shapiro(ssim_kdm)}')
    print(f'psnr_idm {stats.shapiro(psnr_idm)}')
    print(f'ssim_idm {stats.shapiro(ssim_idm)}')
    print(f'psnr_wnet {stats.shapiro(psnr_wnet)}')
    print(f'ssim_wnet {stats.shapiro(ssim_wnet)}')
    print('\n')

    # do wilcoxon signed-rank test (non-parametric paired t-test)
    print('Wilcoxon tests for PSNR:')
    print(f'kdm vs noisy {stats.wilcoxon(psnr_kdm, psnr_noisy, alternative='greater')}')
    print(f'idm vs noisy {stats.wilcoxon(psnr_idm, psnr_noisy, alternative='greater')}')
    print(f'wnet vs noisy {stats.wilcoxon(psnr_wnet, psnr_noisy, alternative='greater')}')
    print(f'kdm vs idm {stats.wilcoxon(psnr_kdm, psnr_idm, alternative='greater')}')
    print(f'kdm vs wnet {stats.wilcoxon(psnr_kdm, psnr_wnet, alternative='greater')}')
    print(f'idm vs wnet {stats.wilcoxon(psnr_idm, psnr_wnet, alternative='greater')}')

    print('Wilcoxon tests for SSIM:')
    print(f'kdm vs noisy {stats.wilcoxon(ssim_kdm, ssim_noisy, alternative='greater')}')
    print(f'idm vs noisy {stats.wilcoxon(ssim_idm, ssim_noisy, alternative='greater')}')
    print(f'wnet vs noisy {stats.wilcoxon(ssim_wnet, ssim_noisy, alternative='greater')}')
    print(f'kdm vs idm {stats.wilcoxon(ssim_kdm, ssim_idm, alternative='greater')}')
    print(f'kdm vs wnet {stats.wilcoxon(ssim_kdm, ssim_wnet, alternative='greater')}')
    print(f'idm vs wnet {stats.wilcoxon(ssim_idm, ssim_wnet, alternative='greater')}')

    print('Wilcoxon test for Huber vs MSE:')
    print(f'wnet psnr {stats.wilcoxon(psnr_wnet_mse, psnr_wnet, alternative='greater')}')
    print(f'wnet ssim {stats.wilcoxon(ssim_wnet_mse, ssim_wnet, alternative='greater')}')
    print(f'kdm psnr {stats.wilcoxon(psnr_kdm_mse, psnr_kdm, alternative='greater')}')
    print(f'kdm ssim {stats.wilcoxon(ssim_kdm_mse, ssim_kdm, alternative='greater')}')

    # do wilcoxon rank sum test (non-parametric unpaired t-test)
    print('Wilcoxon test for T1w vs T2w test results:')
    print(f'psnr {stats.ranksums(psnr_test_t1, psnr_test_t2)}')
    print(f'ssim {stats.ranksums(ssim_test_t1, ssim_test_t2)}')


if __name__ == '__main__':
    main()
