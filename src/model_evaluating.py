import torch

def evaluate(model, dl_test):
    cuda=True
    if cuda:
        # moving model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(device)   # only for testing
        model = model.to(device)
    else:
        device = "cpu"

    with torch.no_grad():
        # Set model to eval mode
        model.eval()

        for batch in dl_test:
            x_batch, y_batch = batch[0], batch[1]
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Forward pass
            outputs = model.forward(x_batch)

            print(f"Prediction: {outputs}")
            print(f"RealValue: {y_batch}")

            _, predicted = torch.max(outputs.data, 1)

            total = y_batch.size(0)
            correct = (predicted == y_batch).sum().item()
            accuracy = correct / total
            print(correct)
            print(accuracy)
            # todo: confusion matrix

