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


skip_num = 0
train_num = 0


def train(model, loader):
    global skip_num, train_num
    model.train()
    for data in loader:
        try:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
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
def test(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            preds = output.argmax(dim=1)
            predictions.append(preds)
            true_labels.append(data.y)

    # Flatten lists of tensors into single tensors
    predictions = torch.cat(predictions).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


print("begin trainning")
train_accuracies = []
test_accuracies = []
for epoch in range(1, 201):
    train(model, train_loader)
    if epoch % 10 == 0:
        train_accuracy = test(model, train_loader)
        test_accuracy = test(model, test_loader)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(
            f'Epoch: {epoch}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, Skip: {skip_num},Train: {train_num}'
        )

plt.figure().set_figwidth(15)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
