# dataset source: https://rajpurkar.github.io/SQuAD-explorer/
from typing import List, Tuple

import torch
import torch.nn as nn
import wandb

import json_utils
from constants import (HIDDEN_SIZE, INPUT_SIZE, MODEL_STATE, OUTPUT_SIZE,
                       all_words, tags)
from model import NeuralNetSmall, get_model_data, save_model
from model_utils import get_data_loader, load_model
from utils import get_training_device

wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}


def main():
    torch_file_path_load: str = 'test_train.pth'

    # Hyperparameters
    batch_size: int = 128
    input_size = len(all_words)
    output_size = len(tags)

    # hidden_size = int(np.mean([input_size, output_size])) // 30
    hidden_size = int(output_size * 1.5)

    learning_rate = 0.001
    num_epoch = 100
    num_workers = 1

    # amount of maximum data sets available
    max_data_set = 442
    step = 20

    device = get_training_device()

    # if a pretrained model exists, the weights get loaded into the model
    model_data = load_model(torch_file_path_load)
    if model_data is not None:
        print('pretrained model found')
        input_size = model_data[INPUT_SIZE]
        output_size = model_data[OUTPUT_SIZE]
        hidden_size = model_data[HIDDEN_SIZE]
        model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_data[MODEL_STATE])
    else:
        model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)

    # prepare json file
    dir_name: str = 'models_1.5_hl'
    json_file_name: str = 'accuracy_loss_data.json'
    json_model_name: str = f'NeuralNetSmall(input:{input_size}, hidden:{hidden_size}, output:{output_size})'

    json_utils.init_accuracy_loss_json_file(
        json_model_name, dir_name, json_file_name
    )
    loss_list: List[Tuple[str, float]] = []

    print('start training')
    for epoch in range(num_epoch):
        print(f'epoch: {epoch + 1}')
        # loss and optimizer
        criterion, optimizer = get_criterion_and_optimizer(
            model, learning_rate)
        current_epoch_average_loss: List[float] = []

        for from_range in range(0, max_data_set, step):

            train_loader = get_data_loader(
                from_range, from_range + step, batch_size, num_workers)

            loss = training_loop(train_loader, model, criterion, optimizer)
            current_epoch_average_loss.append(loss.item())
            print(
                f'from_range: {from_range} to_range: {from_range + step} current_loss: {loss}')

        print(f'\nepoch {epoch + 1}/{num_epoch}, loss={loss.item():4f}\n')

        loss_list.append((f'loss_epoch_{epoch + 1}', loss.item()))

        wandb.log({'loss': loss})
        wandb.watch(model)

        data = get_model_data(model, input_size, output_size,
                              hidden_size, all_words, tags)
        save_model(
            data, f'{dir_name}/small_model_hidden_1.5_of_output_epoch_{epoch + 1}.pth')

    json_utils.update_loss(dir_name, json_file_name, loss_list)

    print('done')


def get_criterion_and_optimizer(model: nn.Module, learning_rate: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer


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


if __name__ == '__main__':
    main()
