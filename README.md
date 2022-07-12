# AutoRes
The code repository in this paper uses Siamese neural network (SNN) for automatic resolution of overlapping peaks in complex GC-MS data to extract meaningful features that are swamped by noise, baseline drift, retention time shifts, and overlapping peaks. We developed a method called AutoRes, which, includes a pSCNN1 model and a pSCNN2 model with the same structure but different inputs. pSCNN1 and pSCNN2 can be used to predict the selective and elution regions of each compound in the overlapping peaks, respectively. The predicted regions are used as inputs to the full rank resolution (FRR) method, which can be easily achieved for the overlapping peaks.
# Package required:
We recommend to use conda and pip.
