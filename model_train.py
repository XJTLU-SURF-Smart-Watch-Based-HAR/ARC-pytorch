"""
@File   :   model_train.py
@Date   :   2023/7/14
@Description    :   This file is for training a model after running the model_input.py.
                    Model and evaluation results will be saved accordingly.
"""

import numpy as np
import torch
from numpy import float32
from sklearn.model_selection import StratifiedKFold, KFold
from src import model_training_core as model_training
from src import model_evaluating
from torch.utils.data import DataLoader, TensorDataset
from src import logging

# Make numpy values easier to read.
np.set_printoptions(edgeitems=2, suppress=True, precision=4)

MACRO_DEVELOPMENT = False
# MACRO_DEVELOPMENT = True

if __name__ == '__main__':
    logging.init()
    data = np.load('temp/sliced-data/feature.npy').astype(float32)
    label = np.load('temp/sliced-data/label.npy')

    if MACRO_DEVELOPMENT:
        n_split = 3
        epochs = 10
        cuda = True
        learning_rate = 0.001
        avg_accuracy = 0
    else:
        n_split = 10
        epochs = 100
        cuda = True
        learning_rate = 0.001
        avg_accuracy = 0

    # balance the number of activities in both train and test sets, via StratifiedKFold
    logging.info(f"Split to {n_split} folds")
    kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=28)

    for i, (train_index, test_index) in enumerate(kf.split(data, label)):
        logging.info(f"\nFold {i}:")
        logging.info(f"\tTrain: index={len(train_index)}")
        logging.info(f"\tTest:  index={len(test_index)}")

        testX_path = 'temp/10-fold-cross-validation/only-one-subject/' + str(i) + '-testX.npy'
        testY_path = 'temp/10-fold-cross-validation/only-one-subject/' + str(i) + '-testY.npy'
        model_path = 'temp/model/' + str(i) + '-model.pth'
        confusion_matrix_path = 'temp/confusion-matrix/' + str(i) + '-confusion-matrix'

        np.save(testX_path, data[test_index])
        np.save(testY_path, label[test_index])

        training_dataset = TensorDataset(torch.from_numpy(data[train_index]), torch.from_numpy(label[train_index]))
        testing_dataset = TensorDataset(torch.from_numpy(data[test_index]), torch.from_numpy(label[test_index]))

        dl_train = DataLoader(training_dataset, batch_size=64, shuffle=True, drop_last=True)
        dl_test = DataLoader(testing_dataset, batch_size=1, shuffle=False, drop_last=False)

        model = model_training.run_train_mlp(dl_train, epochs=epochs, learning_rate=learning_rate, cuda=cuda,
                                             model_path=model_path)
        logging.info(f"Parameters of the model: {sum(p.numel() for p in model.parameters())}")

        classes = ['walking', 'jogging', 'stairs', 'sitting', 'standing', 'typing', 'teeth', 'soup', 'chips', 'pasta',
                   'drinking', 'sandwich', 'kicking', 'catch', 'dribbling', 'writing', 'clapping', 'folding']

        accuracy = model_evaluating.evaluate(model, dl_test, classes, cuda, confusion_matrix_path)
        avg_accuracy += accuracy / n_split

    logging.info(f'\n\nOverall Report:\n\tAverage accuracy = {avg_accuracy}\n\t{model}')
    logging.end()