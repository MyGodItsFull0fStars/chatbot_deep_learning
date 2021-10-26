# dataset source: https://rajpurkar.github.io/SQuAD-explorer/
import json
import os.path as path
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from constants import *
from model import NeuralNetSmall, save_model, load_model, get_model_from_torch_file, get_model_data
from squad import Squad
from utils import (
    bag_of_words,
    get_training_device,
    tokenize,
    get_all_words,
    get_tags,
    get_all_words_dict,
)

SQUAD_FILE_PATH: str = "squad_dataset.json"

all_words = get_all_words()
tags = get_tags()
all_words_dict = get_all_words_dict(all_words)


class TrainData:
    def __init__(self, from_range: int = None, to_range: int = None) -> None:
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


def main():
    file_path_load: str = "test_train.pth"
    file_path_store: str = "test_train.pth"

    # Hyperparameters
    batch_size: int = 16
    input_size = len(all_words)
    output_size = len(tags)

    # hidden_size = int(np.mean([input_size, output_size])) // 30
    # hidden_size = int(output_size * 1.2)
    hidden_size = 0

    learning_rate = 0.001
    num_epoch = 5
    num_workers = 12

    # amount of maximum data sets available
    max_data_set = 442
    data_set_to_range = max_data_set
    step = 10

    device = get_training_device()

    # if a pretrained model exists, the weights get loaded into the model
    model_data = load_model(file_path_load)
    if model_data is not None:
        print("pretrained model found")
        input_size = model_data[INPUT_SIZE]
        output_size = model_data[OUTPUT_SIZE]
        hidden_size = model_data[HIDDEN_SIZE]
        model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_data[MODEL_STATE])
    else:
        model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)

    for epoch in range(num_epoch):

        for from_range in range(0, data_set_to_range, step):
            to_range = (
                from_range + step
                if from_range + step <= data_set_to_range
                else data_set_to_range
            )

            print(
                f"epoch: {epoch + 1}/{num_epoch} -- range: [{from_range}-{to_range}]"
            )

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

            loss = training_loop(train_loader, model, criterion, optimizer)

        print(f"\nepoch {epoch + 1}/{num_epoch}, loss={loss.item():4f}\n")
        data = get_model_data(model, input_size, output_size, hidden_size, all_words, tags)
        save_model(data, f'models_1.2_hl/test_train_1.2_epoch_{epoch + 1}.pth')

    # data = get_model_data(model, input_size, output_size, hidden_size)

    # save_model(data, file_path_store)


def training_loop(train_loader, model, criterion, optimizer):
    device = get_training_device()

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

    return loss





if __name__ == "__main__":
    main()
