# dataset source: https://rajpurkar.github.io/SQuAD-explorer/
import json
from typing import List, Tuple

from torch.utils import data
from utils import (
    tokenize,
    stemming,
    bag_of_words,
    get_sorted_unique_string_list,
    get_training_device,
)
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataset

from model import NeuralNet

import numpy as np

from squad import *
from constants import *

SQUAD_FILE_PATH: str = 'squad_dataset.json'


class TrainData:
    def __init__(self) -> None:

        self.ignore_words = ['?', '!', '.', ',', ';']
        self.all_words = self.get_all_words()
        self.X_y = []
        self.tags = self.get_tags()
        self.X_train = []
        self.y_train = []

        with open(SQUAD_FILE_PATH, 'r') as file:
            squad_json = json.load(file)

            squad = Squad(squad_json)

            for squad_data in squad.data_list:
                tag = squad_data.title
                for paragraph in squad_data.paragraphs:
                    # TODO maybe insert again for better results
                    # self.tags.append(paragraph.context)

                    for qas in paragraph.question_answer_sets:
                        question = tokenize(qas.question)
                        self.X_y.append((question, tag))

            
            self.create_bag_of_words()


    def create_bag_of_words(self):
        all_words_dict = self.get_all_words_dict(self.all_words)

        for (pattern_sentence, tag) in self.X_y:
            bag = bag_of_words(pattern_sentence, all_words_dict)
            self.X_train.append(bag)

            label = self.tags.index(tag)
            self.y_train.append(label)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)


    def get_X_y_train(self):
        return self.X_train, self.y_train


    def get_all_words_dict(self, all_words) -> Dict[str, int]:
        all_words_dict = {}

        for idx, word in enumerate(all_words):
            all_words_dict[word] = idx

        return all_words_dict

    def get_all_words(self) -> List[str]:
        with open('all_words.txt', 'r') as word_file:
            word_str = word_file.read()
            all_words = word_str.split(' ')

            return all_words

    def get_tags(self) -> List[str]:
        with open('tags.txt', 'r') as tags_file:
            tags_str = tags_file.read()
            tags = tags_str.split(' ')

            return tags


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
    train_data = TrainData()

    # Hyperparameters
    batch_size: int = 32
    input_size = len(train_data.all_words)
    hidden_size = 128
    output_size = len(train_data.tags)
    learning_rate = 0.001
    num_epoch = 1000

    X_train, y_train = train_data.get_X_y_train()

    dataset = ChatDataSet(X_train, y_train)

    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=5
    )

    device = get_training_device()
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('Start training')

    # training loop
    for epoch in range(num_epoch):
        for (words, labels) in train_loader:
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

        if (epoch + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epoch}, loss={loss.item():4f}')

    print(f'final loss={loss.item():4f}')

    data = {
        MODEL_STATE: model.state_dict(),
        INPUT_SIZE: input_size,
        OUTPUT_SIZE: output_size,
        HIDDEN_SIZE: hidden_size,
        ALL_WORDS: train_data.all_words,
        TAGS: train_data.tags,
    }

    file_name = 'data.pth'
    torch.save(data, file_name)

    print(f'training complete. file saved to {file_name}')


if __name__ == '__main__':
    # main()
    train = TrainData()
