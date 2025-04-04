# dual-domain-mri-denoising

Benefits such as greater accessibility and decreased costs of lower-field magnets in magnetic resonance imaging have increased the use of low-field scanners. However, a challenge that must be addressed to produce clinically useful images is the noise and lack of signal associated with decreased magnetic field strengths. Dual-domain neural networks, such as WNet, show promise in medical image denoising. Results of comparing k-space-denoising, image-denoising, and dual-domain denoising model performance on T1-weighted images show best peak signal-to-noise ratio and structural similarity with a stand-alone k-space denoiser. Training on T1-weighted images also provides sufficient denoising with T2-weighted images, supporting model viability with multi-contrast acquisitions. Future experiments should focus on improving preservation of finer structural details in both gray and white brain matter to address issues with blurring artifacts, and validation on more realistic noise sources including radiofrequency bands caused by interference in the scanner environment.

*Keywords*: magnetic resonance imaging, image denoising, dual-domain 

Original WNet:
- Cheslerean-Boghiu, T., Hofmann, F. C., Schultheiß, M., Pfeiffer, F., Pfeiffer, D., & Lasser, T. (2023). 
WNet: A Data-Driven Dual-Domain Denoising Model for Sparse-View Computed Tomography With a Trainable Reconstruction Layer. *IEEE Transactions on Computational Imaging, 9*, 120-132. https://doi.org/10.1109/TCI.2023.3240078 

## Dataset

This project uses multicoil brain MRI data from the NYU fastMRI dataset:

- Knoll, F., Zbontar, J., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., Parente, M., Geras, K. J., Katsnelson, J., Chandarana, H., Zhang, Z., Drozdzalv, M., Romero, A., Rabbat, M., Vincent, P., Pinkerton, J., Wang, D., Yakubova, N., Owens, E., Zitnick, C. L., Recht, M. P., Sodickson, D. K., & Lui, Y. W. (2020). fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning. *Radiology: Artificial Intelligence, 2*(1). https://doi.org/10.1148/ryai.2020190007 
- Zbontar, J. Knoll, F., Sriram, A., Murrell, T., Huang, Z., Muckley, M. J., Defazio, A., Stern, R., Johnson, P., Bruno, M., Parente, M., Geras, K. J., Katnelson, J., Chandarana, H., Zhang, Z., Drozdzal, M., Romero, A., Rabbat, M., Vincent, P., Yakubova, N., Pinkerton, J., Wang, D., Owens, E., Zitnick, C. L., Recht, M. P., Sodickson, D. K., & Lui, Y. W. (2019). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. *arXiv*, 1811.08839. https://doi.org/10.48550/arXiv.1811.08839 

The exact subset of images used for training/validation/testing are listed in `constants.py`.

## Additional Notes
- Training can be reproduced by running `data_prep.py` > `training.py` > `inference.py` with input images in the `data` folder
- Pickle and .h5 files are not available due to large file sizes