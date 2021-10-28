import json
import os
from copy import deepcopy
from typing import List, Tuple, ValuesView

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from constants import *
from model import NeuralNetSmall
from squad import Squad
from utils import bag_of_words, get_training_device, tokenize

device = get_training_device()

# amount of maximum data sets available
max_data_set: int = 442

SQUAD_FILE_PATH: str = 'squad_dataset.json'


class TrainData:
    def __init__(self, from_range: int = None, to_range: int = None) -> None:
        self.all_words = all_words
        self.X_y = []
        self.tags = tags
        self.X_train = []
        self.y_train = []

        self.from_range = from_range if from_range is not None else 0

        with open(SQUAD_FILE_PATH, 'r') as file:
            squad_json = json.load(file)
            squad = Squad(squad_json)

            self.to_range = to_range if to_range is not None else len(
                squad.data_list)

            self.init_X_y_set(squad)
            self.create_bag_of_words()

    def init_X_y_set(self, squad: Squad) -> None:
        for squad_data in squad.data_list[self.from_range: self.to_range]:
            tag = squad_data.title

            for paragraph in squad_data.paragraphs:

                for qas in paragraph.question_answer_sets:
                    question = tokenize(qas.question)
                    self.X_y.append((question, tag))

    def create_bag_of_words(self):
        for (pattern_sentence, tag) in self.X_y:

            bag = bag_of_words(pattern_sentence, all_words_dict)

            self.X_train.append(bag)

            label = self.tags.index(tag)
            self.y_train.append(label)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

    def get_X_y_train(self):
        return self.X_train, self.y_train


class ChatDataSet(Dataset):
    def __init__(self, X_train, y_train) -> None:
        super().__init__()
        self.n_samples = len(X_train)
        self.X_data_ = deepcopy(X_train)
        self.y_data_ = deepcopy(y_train)

    # dataset[idx]
    def __getitem__(self, index) -> Tuple:
        return self.X_data_[index], self.y_data_[index]

    def __len__(self) -> int:
        return self.n_samples


def get_accuracy(train_loader: DataLoader, model: Module) -> Tuple[int, int]:
    "This function retuns a tuple with the accuracy of the model and the correct and total labels"
    # disables gradient calculations
    if train_loader is None or model is None:
        raise ValueError('invalid parameters')

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
            model = get_model_from_torch_file(model_file_name)
        else:
            print(f'model not found in directory: {dir_name}')
    elif os.path.exists(model_file_name):
        model = get_model_from_torch_file(model_file_name)
    else:
        print(f'file name does not exist')

    return model


def get_model_from_torch_file(model_file_name: str) -> Module:
    model_data = load_model(model_file_name)
    if model_data is not None:
        input_size = model_data[INPUT_SIZE]
        output_size = model_data[OUTPUT_SIZE]
        hidden_size = model_data[HIDDEN_SIZE]
        model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_data[MODEL_STATE])

        return model
    else:
        raise ValueError('invalid file name, cannot load torch file')


def save_model(data: dict, file_name: str = "data.pth"):
    torch.save(data, file_name)
    print(f"training complete. file saved to {file_name}")


def load_model(file_name: str = "data.pth"):
    if os.path.exists(file_name):
        return torch.load(file_name)
    else:
        return None


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
