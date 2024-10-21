import matplotlib.pyplot as plt
from typing import List


def draw_accuracy(epochs: List[int], train_accuracies: List[float], test_accuracies: List[float]):
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


def draw_loss(epochs: List[int], train_losses: List[float], test_losses: List[float]):
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
    plt.close()
