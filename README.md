## AutoRes

This is the code repo for the paper *Deep Learning-based Method for Automatic Resolution of GC-MS Data from Complex Samples*. We proposed an Automatic Resolution method (AutoRes) for overlapped peaks in complex GC-MS data based on the pseudo-Siamese convolutional neural network (pSCNN) architecture. It consists of two pSCNN models (pSCNN1 and pSCNN2) with the same architecture but different inputs. Two pSCNN models were trained with 400,000 augmented spectral pairs, respectively. They can predict the selective region (pSCNN1) and elution region (pSCNN2) of compounds in an untargeted manner. The predicted regions are used as inputs to the full rank resolution (FRR) method, which can be easily achieved for the overlapping peaks.

<div align="center">
<img src="https://github.com/dyjfan/AutoRes/blob/main/workflow.png" width=809 height=970 />
</div>






### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)
- [tensorflow](https://www.tensorflow.org) 

The main packages can be seen in [requirements.txt](https://github.com/dyjfan/AutoRes/blob/main/requirements.txt).

## Data augmentation

The mass spectral pairs of the training pSCNN1 and pSCNN2 models are obtained using the [data_augmentation_1](https://github.com/dyjfan/AutoRes/blob/main/pSCNN/da.py#L23) and [data_augmentation_2](https://github.com/dyjfan/AutoRes/blob/main/pSCNN/da.py#L47) functions.

    aug_eval1 = data_augmentation_1(spectra, n, maxn, noise_level=0.001)
    aug_eval2 = data_augmentation_2(spectra, c, n, m, maxn, noise_level=0.001)

*Optionnal args*
- spectra : Mass spectral library 
- c : imilar sublibrary
- n ：Number of amplified mass spectral pairs
- m : Number of amplified mass spectral pairs with high similarity.
- maxn ：Number of components

## Model training
Train the model based on your own training dataset with [build_pSCNN](https://github.com/dyjfan/AutoRes/blob/main/pSCNN/snn.py#L69) function.

    model = build_pSCNN(para)

*Optionnal args*
- para : Hyperparameters for model training

## Automatic Resolution

Automatic Resolution of GC-MS data files by using the [AutoRes](https://github.com/dyjfan/AutoRes/blob/main/AutoRes/AutoRes.py#L354) function.

    AutoRes(ncr, model1, model2)
    
*Optionnal args*
- ncr : GC-MS data
- model1 : pSCNN1 model
- model2 : pSCNN2 model

## Clone the repository and run it directly
[git clone](https://github.com/dyjfan/AutoRes)

An example has been provided in [test.ipynb](https://github.com/dyjfan/AutoRes/blob/main/test.ipynb) script for the convenience of users.Users can run it directly by placing the unzipped data file and AutoRes-1.0 file in the same directory after downloading.

## Contact
- fanyingjie@csu.edu.cn
