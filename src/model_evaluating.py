"""
@File   :   model_evaluating.py
@Date   :   2023/7/14
@Description    :   This file is for evaluating the model via test data
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from src import logging


def plot_confusion_matrix(cm, classes, description, confusion_matrix_path):
    plt.figure(figsize=(8, 6), dpi=480)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add accuracy text below the plot
    plt.text(0.5, -0.3, description, horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)

    plt.tight_layout()
    # plt.savefig(confusion_matrix_path+'.pdf', format='pdf')
    plt.savefig(confusion_matrix_path + '.jpg', format='jpeg')


def evaluate(model, dl_test, class_names, cuda, confusion_matrix_path):
    if cuda:
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        device = "cpu"

    with torch.no_grad():
        # Set model to eval mode
        model.eval()

        true_labels = []
        predicted_labels = []
        correct = 0

        for batch in dl_test:
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model.forward(x_batch)

            _, predicted = torch.max(outputs.data, 1)

            if predicted == y_batch:
                correct += 1

            # Collect true labels and predicted labels for confusion matrix
            true_labels.extend(y_batch.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

        total = len(dl_test)
        accuracy = correct / total * 100
        description = f'NB: Accuracy = Correct / Total = {correct} / {total} = {accuracy:.2f}%'
        logging.info(description)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot confusion matrix
        plot_confusion_matrix(cm, classes=class_names, description=description,
                              confusion_matrix_path=confusion_matrix_path)
        # plt.show()

    return accuracy
