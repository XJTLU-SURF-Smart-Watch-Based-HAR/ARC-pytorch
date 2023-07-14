"""
@File   :   feature_extracting.py
@Date   :   2023/7/14
@Description    :   This file is for extracting the features from windows
"""

import numpy as np
import pycatch22


def extract(windows, featurize):
    """
    This is the core function for extracting, you could specify which extracting strategy to be used while calling
    @param windows:
    @param featurize:
    @return:
    """
    features = []  # FOR ALL WINDOWS

    for window in windows:
        feature = []  # FOR ALL AXIS IN ONE WINDOW
        for axis in window:
            flat_data = [float(item) for item in axis]
            feature.extend(featurize(flat_data))
        features.append(feature)
    return np.array(features)


def catch22(data) -> list:
    return pycatch22.catch22_all(data, catch24=True)['values']

# define the new extracting strategy as below
# def new_strategy(data) -> list:
    # pass
