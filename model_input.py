"""
@File   :   model_input.py
@Date   :   2023/7/14
@Description    :   This file is for generating the input data for model_train.py, namely features $ labels.
"""

from src import data_loading
from src import feature_extracting

import numpy as np

# Make numpy values easier to read.
np.set_printoptions(edgeitems=2, suppress=True, precision=4)

if __name__ == '__main__':
    txt_filepath = 'data/WISDM-Dataset-Watch-Gyro-Raw/data_1600_gyro_watch.txt'
    ten_folds_filepath = 'temp/10-fold-cross-validation/only-one-subject'
    sliced_filepath = 'temp/sliced-data/1600.txt'

    print(f"\n1. Reading from \"{txt_filepath}\":")
    data, label = data_loading.txt2sliced(txt_filepath, sliced_filepath)
    print(f"\t#windows={len(data)}, #labels={len(label)}\n")

    print("2. Feature extracting:")
    feature = feature_extracting.extract(windows=data, featurize=feature_extracting.catch22)
    print(f"\t#windows={len(feature)}, #features per window={len(feature[0])}\n{feature}\n\n{label}")

    np.save('temp/sliced-data/feature.npy', feature)
    np.save('temp/sliced-data/label.npy', label)
