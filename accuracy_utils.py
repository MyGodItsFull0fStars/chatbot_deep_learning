from typing import Tuple
import torch
from torch.nn import Module

from torch.utils.data import DataLoader

from utils import get_training_device

device = get_training_device()

def get_accuracy(train_loader: DataLoader, model: Module) -> Tuple[float, int, int]:
    "This function retuns a tuple with the accuracy of the model and the correct and total labels"
    # disables gradient calculations
    with torch.no_grad():
        correct = 0
        total = 0

        for words, labels in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (correct/total, correct, total)
