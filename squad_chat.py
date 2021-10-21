import json

import torch

from constants import *
from model import NeuralNetSmall
from utils import bag_of_words, get_all_words_dict, get_training_device, tokenize

device = get_training_device()
bot_name = 'Alan'


def bot_answer(answer: str) -> None:
    print(f'{bot_name}: {answer}')


def chat_loop(chat_model) -> None:
    print("Hey lets chat! Type 'quit' to exit")

    while True:

        sentence = input('You: ')

        if sentence == 'quit':
            bot_answer('Goodbye ;)')
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words_dict)
        # 1 row because we have one sample
        # X.shape[0] number of columns
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = chat_model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        print(prob)

        if prob.item() > 0.01:
            print(intents.keys())
            for intent in intents[INTENTS]:
                if tag == intent[TAG]:
                    bot_answer(intent[0])
        else:
            bot_answer('I do not understand...')


with open('squad_dataset.json', 'r') as file:
    intents = json.load(file)

    torch_file_path = 'test_train_5_episodes.pth'
    data = torch.load(torch_file_path)

    input_size = data[INPUT_SIZE]
    hidden_size = data[HIDDEN_SIZE]
    output_size = data[OUTPUT_SIZE]
    all_words = data[ALL_WORDS]
    all_words_dict = get_all_words_dict(all_words)
    tags = data[TAGS]
    model_state = data[MODEL_STATE]

    model = NeuralNetSmall(input_size, hidden_size, output_size).to(device)

    # load the learned parameters
    model.load_state_dict(model_state)

    # set model to evaluation mode
    model.eval()

    chat_loop(model)
