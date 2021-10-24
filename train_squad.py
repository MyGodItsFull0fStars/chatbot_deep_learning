# dataset source: https://rajpurkar.github.io/SQuAD-explorer/
import json
import os.path as path
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from constants import *
from model import NeuralNetSmall
from squad import Squad
from utils import (bag_of_words, get_training_device, tokenize,
                   get_all_words, get_tags, get_all_words_dict)

SQUAD_FILE_PATH: str = "squad_dataset.json"

all_words = get_all_words()
tags = get_tags()


class TrainData:

    def __init__(self, from_range: int = None, to_range: int = None) -> None:
        print("init TrainData")

        self.all_words = all_words
        self.X_y = []
        self.tags = tags
        self.X_train = []
        self.y_train = []

        self.from_range = from_range if from_range is not None else 0

        with open(SQUAD_FILE_PATH, "r") as file:
            squad_json = json.load(file)
            squad = Squad(squad_json)

            self.to_range = to_range if to_range is not None else len(
                squad.data_list)

            self.init_X_y_set(squad)
            self.create_bag_of_words()

    def init_X_y_set(self, squad: Squad) -> None:
        print("init X_y set")
        for squad_data in squad.data_list[self.from_range: self.to_range]:
            tag = squad_data.title

            for paragraph in squad_data.paragraphs:

                for qas in paragraph.question_answer_sets:
                    question = tokenize(qas.question)
                    self.X_y.append((question, tag))

    def create_bag_of_words(self):
        print("create bag of words")
        all_words_dict = get_all_words_dict(self.all_words)

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


def main():
    file_name: str = "test_train.pth"

    # Hyperparameters
    batch_size: int = 16
    input_size = len(all_words)
    output_size = len(tags)

    # hidden_size = int(np.mean([input_size, output_size])) // 30
    hidden_size = int(output_size * 1.2)

    learning_rate = 0.001
    num_epoch = 5
    num_workers = 12

    # amount of maximum data sets available
    max_data_set = 442
    data_set_to_range = max_data_set
    step = 10

    device = get_training_device()

    model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)
    # if a pretrained model exists, the weights get loaded into the model
    model_data = load_model(file_name)
    if model_data is not None:
        model.load_state_dict(model_data[MODEL_STATE])

    for epoch in range(num_epoch):

        for from_range in range(0, data_set_to_range, step):
            to_range = (
                from_range + step
                if from_range + step <= data_set_to_range
                else data_set_to_range
            )

            print(f"epoch: {epoch} -- range: [{from_range}--{to_range}]")

            train_data = TrainData(from_range, to_range)
            X_train, y_train = train_data.get_X_y_train()
            dataset = ChatDataSet(X_train, y_train)

            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            # loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            loss = training_loop(1, train_loader, model, criterion, optimizer)

        print(f"\nepoch {epoch + 1}/{num_epoch}, loss={loss.item():4f}\n")

    data = get_model_data(model, input_size, output_size, hidden_size)
    
    save_model(data, file_name)


def training_loop(num_epoch: int, train_loader, model, criterion, optimizer):
    print("Start training")
    device = get_training_device()

    for _ in range(num_epoch):
        count = 0
        for (words, labels) in train_loader:
            # Move tensors to the configured device
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # forward
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # backward and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"current count: {count}")
            count += 1

    return loss


def save_model(data: dict, file_name: str = "data.pth"):
    torch.save(data, file_name)
    print(f"training complete. file saved to {file_name}")


def load_model(file_name: str = "data.pth"):
    if path.exists(file_name):
        return torch.load(file_name)
    else:
        return None


def get_model_from_torch_file(torch_file)  -> dict:
    data_dict: dict = {
        key: torch_file[key] for key in torch_file.keys()
    }

    return data_dict


def get_model_data(
    model, input_size, output_size, hidden_size) -> dict:
    data = {
        MODEL_STATE: model.state_dict(),
        INPUT_SIZE: input_size,
        OUTPUT_SIZE: output_size,
        HIDDEN_SIZE: hidden_size,
        ALL_WORDS: all_words,
        TAGS: tags,
    }
    return data


if __name__ == "__main__":
    main()
