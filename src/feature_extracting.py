import numpy as np
import pycatch22


def extract(windows, featurize) -> list:
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
