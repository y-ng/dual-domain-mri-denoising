# dual-domain-mri-denoising

TODO: Insert abstract here

## Dataset

This project uses multicoil brain MRI data from the NYU fastMRI dataset:

- Knoll, F., Zbontar, J., Sriram, A., Muckley, M. J., Bruno, M., Defazio, A., Parente, M., Geras, K. J., Katsnelson, J., Chandarana, H., Zhang, Z., Drozdzalv, M., Romero, A., Rabbat, M., Vincent, P., Pinkerton, J., Wang, D., Yakubova, N., Owens, E., Zitnick, C. L., Recht, M. P., Sodickson, D. K., & Lui, Y. W. (2020). fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images for Accelerated MR Image Reconstruction Using Machine Learning. *Radiology: Artificial Intelligence, 2*(1). https://doi.org/10.1148/ryai.2020190007 
- Zbontar, J. Knoll, F., Sriram, A., Murrell, T., Huang, Z., Muckley, M. J., Defazio, A., Stern, R., Johnson, P., Bruno, M., Parente, M., Geras, K. J., Katnelson, J., Chandarana, H., Zhang, Z., Drozdzal, M., Romero, A., Rabbat, M., Vincent, P., Yakubova, N., Pinkerton, J., Wang, D., Owens, E., Zitnick, C. L., Recht, M. P., Sodickson, D. K., & Lui, Y. W. (2019). fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. *arXiv*, 1811.08839. https://doi.org/10.48550/arXiv.1811.08839 

The exact subset of images used for training/validation/testing are listed in `constants.py`.

## Additional Notes
- Training can be reproduced by running `data_prep.py` > `training.py` > `inference.py` with input images in the `data` folder
- Pickle and .h5 files are not available due to large file sizes