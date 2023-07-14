# Description

This repository aims to separate the data input and model training processes. The first part primarily includes packages required for feature extraction, such as pycatch22, while the second part only includes PyTorch and CUDA itself, for model training.

Currently, the data input process does not utilize PyTorch's recommended Custom Dataset. Instead, it focuses on data handling, feature generation, and saving them as .npy files, preparing inputs for model training.

# How to use
1. You have to create two conda environments to separately run the model_input.py & model_train.py.

Possible CLI:
To run model_input.py
```commandline
conda env create --name model_input_related
conda activate model_input_related
conda install -c conda-forge pycatch22
...(numPY, e.t.c)
```

To run model_train.py
```commandline
conda env create --name model_train_related
conda activate model_train_related
conda install -c conda-forge pytorch
...(numPY, matplotlib, sklearn, e.t.c.)
```

# Repository Structure

```
├── data
│   ├── WISDM-Dataset-Activity-Key.txt
│   ├── WISDM-Dataset-Description.pdf
│   └── WISDM-Dataset-Watch-Gyro-Raw
│       └── data_1600_gyro_watch.txt    #Download from [UCI](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
├── logs
│   └── HAR.log
├── model_input.py      #Main file to generate input data for data training
├── model_train.py      #Main file to train & evaluate a model after running model_input.py
├── README.md
├── runConfigurations
│   ├── model_input.run.xml    
│   └── model_train.run.xml
├── src
│   ├── data_loading.py
│   ├── feature_extracting.py
│   ├── logging.py
│   ├── model_evaluating.py
│   ├── models.py
│   └── model_training_core.py
└── temp
    ├── 10-fold-cross-validation
    │   ├── leave-one-subject
    │   └── only-one-subject    #For the moment, this project only supports one-subject training
    │      ├── 0-testX.npy
    │      └── 0-testY.npy
    ├── confusion-matrix
    │   └── 0-confusion-matrix.jpg
    ├── model
    │   └── 0-model.pth
    └── sliced-data
        ├── 1600.txt
        ├── feature.npy
        └── label.npy
```


