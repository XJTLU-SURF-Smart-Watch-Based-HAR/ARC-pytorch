import numpy as np
import torch
from numpy import float32
from sklearn.model_selection import StratifiedKFold, KFold
from src import model_training_core as model_training
from src import model_evaluating
from torch.utils.data import DataLoader, TensorDataset

# Make numpy values easier to read.
np.set_printoptions(edgeitems=2, suppress=True, precision=4)

# MACRO_DEVELOPMENT = False
MACRO_DEVELOPMENT = True

if __name__ == '__main__':
    data = np.load('temp/sliced-data/feature.npy').astype(float32)
    label = np.load('temp/sliced-data/label.npy')

    if MACRO_DEVELOPMENT:
        n_split = 3
    else:
        n_split = 10

    print(f"3. Split to {n_split} folds")
    # balance the number of activities in both train and test sets, via StratifiedKFold
    kf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=28)

    for i, (train_index, test_index) in enumerate(kf.split(data, label)):
        print(f"\nFold {i}:")
        print(f"  Train: index={len(train_index)}")
        print(f"  Test:  index={len(test_index)}")

        np.save('temp/10-fold-cross-validation/only-one-subject/'+str(i)+'-testX.npy', data[test_index])
        np.save('temp/10-fold-cross-validation/only-one-subject/'+str(i)+'-testY.npy', label[test_index])

        training_dataset = TensorDataset(torch.from_numpy(data[train_index]), torch.from_numpy(label[train_index]))
        testing_dataset = TensorDataset(torch.from_numpy(data[test_index]), torch.from_numpy(label[test_index]))

        dl_train = DataLoader(training_dataset, batch_size=256, shuffle=True, drop_last=True)
        dl_test = DataLoader(testing_dataset, batch_size=1, shuffle=False, drop_last=False)

        model = model_training.run_train_mlp(dl_train, epochs=10, learning_rate=0.01, cuda=True)
        print(sum(p.numel() for p in model.parameters()))

        model_evaluating.evaluate(model, dl_test)

        exit(0)
