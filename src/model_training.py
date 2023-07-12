import torch
import torch.nn as nn
import torch.optim as optim

# from src import helpers
from src import models


def train_mlp(model, dl_train, epochs, learning_rate, cuda):
    train_losses = []
    train_acc = []

    if cuda:
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        device = "cpu"

    loss = nn.CrossEntropyLoss()  # Classification => Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train Model
    print("Start of training ...")
    for epoch in range(epochs):

        # Training
        model.train()
        running_loss = 0.0
        train_accuracy = 0.0

        for batch in dl_train:
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()  # reset gradients to avoid incorrect calculation
            prediction = model.forward(x_batch)
            l = loss(prediction, y_batch)
            l.backward()
            optimizer.step()
            running_loss += l.item()

            # Accuracy
            top_p, top_class = torch.exp(prediction).topk(1, dim=1)
            equals = top_class == y_batch.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Save loss and accuracy for training
        train_losses.append(running_loss
                            / len(dl_train))
        train_acc.append((train_accuracy / len(dl_train)) * 100)

        # Output current status on console
        print("Epoch: {:03d}/{:03d}".format(epoch + 1, epochs),
              "Training loss: {:.3f}".format(running_loss / len(dl_train)),
              "Training Accuracy: {:.3f}".format((train_accuracy / len(dl_train)) * 100))

    print("Training completed!")

    # plotting
    # if (plots == True):
    # helpers.plot_loss_and_acc(epochs, train_losses, train_acc)

    return model, train_losses


def evaluation(model, cuda, dl_test):
    """This function performs the evaluation of the model with the test data set."""
    # Input:
    # model; the pytorch trained model

    if cuda:
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    # monitoring - evaluate test loss
    print("Start of validation ...")
    with torch.no_grad():  # no gradients, because just monitoring, no optimization

        model.eval()  # Set the model to evaluation mode
        old_id = 0
        bestpred = []
        for batch in dl_test:

            new_id = int(batch[0, 0].item())
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            prediction = model(x_batch)
            top_p, top_class = torch.exp(prediction).topk(1, dim=1)
            top_class = top_class.long()

            # Check sample_id and write value in txt if changed
            if old_id == new_id:
                if not bestpred:
                    bestpred = top_class
                else:
                    bestpred = torch.cat((bestpred, top_class))
            else:
                bestpred = top_class

            old_id = new_id

    print("Validation completed!")


# def testdifhyperparameter():
#     """This funciton trys different values of the hyperparameter (user parameters) settings."""
#
#     base_model_list = [models.MLP_NET_V1()]
#     batch_size = [16, 64, 128, 256, 512]  # Batch size
#     learning_rate = [0.01, 0.001, 0.0001]  # Learning rate
#
#     # Test different models, batch sizes and learning rates
#     for base_model in base_model_list:  # different models
#         for ba in batch_size:  # different batch sizes
#             for lr in learning_rate:  # different learning rates
#                 print("Model:", base_model.__class__.__name__, " |  Optimizer: Adam  |  Batch size:", ba,
#                       " |  Learning rate:", lr)
#                 model, losses = train_mlp(base_model, 50, ba, lr, True, True)
#                 print(model)
#                 evaluation(model)
#                 print()


def run_train_mlp(dl_train, dl_test, epochs=100, learning_rate=0.001, cuda=True, plots=True):
    """This function performs the training and validation for the MLPm CNN V1 and GRU."""
    # The model, hyperparameters and other settings can be changed directly in this function.

    mlp_v1_model = models.MLP()

    # train the model
    model, train_losses = train_mlp(mlp_v1_model, dl_train, epochs, learning_rate, cuda)

    # only for testing
    # print(train_losses)
    # print("Parameters of the model:", helpers.count_parameters_of_model(model))
    torch.save(model, "temp/model/model.pth")
    model = torch.load("temp/model/model.pth")
    # testdifhyperparameter()

    # evaluate the model
    evaluation(model, cuda, dl_test)
