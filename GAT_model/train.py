from os import path
import sys

root_dir = path.abspath(path.dirname(path.dirname(__file__)))
sys.path.append(root_dir)

import torch
import gc
from dataset import *
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from model import *
from typing import Tuple


def get_dataset() -> MyDataset:
    data = MyDataset(root="./")
    # size = len(data)
    # indices = list(range(int(size * 0.6)))
    # return data.index_select(indices)
    return data


def train(model, loader) -> float:
    model.train()
    loss_total = 0
    for data in loader:
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        except Exception as e:
            # print(e)
            continue
        gc.collect()
        torch.cuda.empty_cache()


# Testing Loop
def test(model, loader) -> Tuple[float, float]:
    loss_total = 0
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = criterion(output, data.y)
            loss_total += loss.item()
            preds = output.argmax(dim=1)
            predictions.append(preds)
            true_labels.append(data.y)

    # Flatten lists of tensors into single tensors
    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    accuracy = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    # Store metrics in a dictionary
    metrics = {'accuracy': accuracy, 'loss': loss_total / len(loader), 'recall': recall, 'f1_score': f1}

    return metrics


def train_test(model, train_loader, test_loader, epoch_times: int, test_step: int, draw_data=True):
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    epochs = []
    print(f"{'Epoch':<10}{'Train Acc':<15}{'Test Acc':<15}{'Train Loss':<15}{'Test Loss':<15}{'Test Recall':<15}{'Test F1':<15}")
    for epoch in range(1, epoch_times + 1):
        train(model, train_loader)
        if epoch % test_step == 0:
            train_metrics = test(model, train_loader)
            test_metrics = test(model, test_loader)

            train_accuracies.append(train_metrics['accuracy'])
            test_accuracies.append(test_metrics['accuracy'])
            epochs.append(epoch)
            train_losses.append(train_metrics['loss'])
            test_losses.append(test_metrics['loss'])

            # print(f"{epoch:<10}{train_accuracy:<15.4f}{test_accuracy:<15.4f}{train_loss:<15.4f}{test_loss:<15.4f}")
            print(
                f"{epoch:<10}{train_metrics['accuracy']:<15.4f}{test_metrics['accuracy']:<15.4f}{train_metrics['loss']:<15.4f}{test_metrics['loss']:<15.4f}{test_metrics['recall']:<15.4f}{test_metrics['f1_score']:<15.4f}"
            )

    avg_train_accu = sum(train_accuracies) / len(train_accuracies)
    avg_test_accu = sum(test_accuracies) / len(test_accuracies)
    print(f"avg train accuracy: {avg_train_accu:.4f}, avg test accuracy: {avg_test_accu:.4f}, min test loss: {min(test_losses):4f}")

    if draw_data:
        from draw import draw_loss, draw_accuracy

        draw_loss(epochs, train_losses, test_losses)
        draw_accuracy(epochs, train_accuracies, test_accuracies)

    return avg_test_accu


if __name__ == "__main__":
    dataset = get_dataset()

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=55)
    print("train size: ", len(train_indices))
    print("test size: ", len(test_indices))
    print(f'Number of features: {dataset.num_features}')

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    parameters = {"batch_size": 16, "dropout": 0.25, "learning_rate": 0.001, "epoch_times": 100, "test_step": 10}

    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=parameters["batch_size"], shuffle=False, drop_last=True)

    model = GATModel(in_channels=31, out_channels=2, dropout=parameters["dropout"])
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss() # for in_channels=1

    print("begin training, parameters: ", parameters)
    train_test(
        model,
        train_loader,
        test_loader,
        parameters["epoch_times"],
        parameters["test_step"],
    )
    print("finish training")
