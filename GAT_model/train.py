from os import path
import sys

root_dir = path.abspath(path.dirname(path.dirname(__file__)))
sys.path.append(root_dir)

import random
import matplotlib.pyplot as plt
import torch
import gc
from dataset import MyDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import GATModel
from typing import Tuple


def get_dataset() -> MyDataset:
    data = MyDataset(root="./")
    # size = len(data)
    # indices = list(range(int(size * 0.6)))
    # return data.index_select(indices)
    return data


dataset = get_dataset()

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
print("train size: ", len(train_indices))
print("test size: ", len(test_indices))
print(f'Number of features: {dataset.num_features}')
print(f'Number of first graph edges: {dataset[0].edge_index._indices().shape[1]}')

train_dataset = dataset[train_indices]
test_dataset = dataset[test_indices]

BATCH_SIZE = 16
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

model = GATModel(in_channels=15, out_channels=2)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")
#     model = torch.nn.DataParallel(model)
#     model.to(device)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss() # for in_channels=1


skip_num = 0
train_num = 0


def train(model, loader) -> float:
    global skip_num, train_num
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
            train_num += 1
        except Exception as e:
            # print(e)
            skip_num += 1
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

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy, loss_total / len(loader)


print("begin trainning")
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []
epochs = []
print(f"{'Epoch':<10}{'Train Acc':<15}{'Test Acc':<15}{'Train Loss':<15}{'Test Loss':<15}{'Skip':<10}")
for epoch in range(1, 201):
    train(model, train_loader)
    if epoch % 10 == 0:
        train_accuracy, train_loss = test(model, train_loader)
        test_accuracy, test_loss = test(model, test_loader)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"{epoch:<10}{train_accuracy:<15.4f}{test_accuracy:<15.4f}{train_loss:<15.4f}{test_loss:<15.4f}{skip_num:<10}")

plt.figure().set_figwidth(15)
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, test_accuracies, label='Test Accuracy', marker='o')
for i in range(len(epochs)):
    plt.text(epochs[i], train_accuracies[i], f'{train_accuracies[i]:.2f}', ha='center', va='bottom')
    plt.text(epochs[i], test_accuracies[i], f'{test_accuracies[i]:.2f}', ha='center', va='bottom')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.savefig("accuracy.png")
plt.close()


plt.figure().set_figwidth(15)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
# 在每个数据点添加文本
for i in range(len(epochs)):
    plt.text(epochs[i], train_losses[i], f'{train_losses[i]:.2f}', ha='center', va='bottom')
    plt.text(epochs[i], test_losses[i], f'{test_losses[i]:.2f}', ha='center', va='bottom')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.savefig("loss.png")
