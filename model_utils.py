import os
from typing import List, Tuple
import torch
from torch.nn import Module

from torch.utils.data import DataLoader, dataset
from train_squad import ChatDataSet, TrainData

from utils import get_training_device

device = get_training_device()

# amount of maximum data sets available
max_data_set: int = 442


def get_accuracy(train_loader: DataLoader, model: Module) -> Tuple[int, int]:
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


def get_model(dir_name: str, model_file_name: str) -> Module:
    model: Module = None
    if dir_name is not None and os.path.exists(dir_name):
        os.chdir(dir_name)
        if os.path.exists(model_file_name):
            return torch.load(model_file_name)
        else:
            print(f'model not found in directory: {dir_name}')
    elif os.path.exists(model_file_name):
        return torch.load(model_file_name)
    else:
        print(f'file name does not exist')

    return model


def save_model(data: dict, file_name: str = "data.pth"):
    torch.save(data, file_name)
    print(f"training complete. file saved to {file_name}")


def get_data_loader(train_data_from_range: int, train_data_to_range: int, batch_size: int, num_workers: int) -> DataLoader:
    train_data_to_range = (
        train_data_to_range
        if train_data_to_range <= max_data_set
        else max_data_set
    )

    train_data: TrainData = TrainData(
        train_data_from_range, train_data_to_range)
    X_train, y_train = train_data.get_X_y_train()

    train_loader = DataLoader(
        dataset=ChatDataSet(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader
