import json

import torch

from copy import copy

from fuzzywuzzy import fuzz

from constants import *
from model import NeuralNetSmall
from squad import Question_Answer_Set, Squad, Squad_Transform
from utils import bag_of_words, get_all_words_dict, get_training_device, tokenize

device = get_training_device()
bot_name = 'chattypedia'


def bot_answer(answer: str) -> None:
    print(f'{bot_name}: {answer}')


def chat_loop(chat_model) -> None:
    bot_answer("Hey lets chat! Type 'quit' to exit")

    while True:

        sentence = input('You: ')


        if sentence == 'quit':
            bot_answer('Goodbye ;)')
            break

        sentence_fuzzy = copy(sentence)
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words_dict)
        # 1 row because we have one sample
        # X.shape[0] number of columns
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = chat_model(X)
        _, predicted = torch.max(output, dim=1)
        title = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.01:
            question_answer_sets = squad_transform.get_question_answer_set(title)
        
            best_ratio = 0
            best_question_answer_set: Question_Answer_Set = None

            for qas in question_answer_sets:
                current_ratio = fuzz.ratio(sentence_fuzzy, qas.question)
                if current_ratio > best_ratio:
                    best_question_answer_set = qas
                    best_ratio = current_ratio


            bot_answer(best_question_answer_set.answer)
            
        else:
            bot_answer('I do not understand...')


with open('squad_dataset.json', 'r') as file:
    squad_file = json.load(file)
    squad_transform: Squad_Transform = Squad_Transform(Squad(squad_file))

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
