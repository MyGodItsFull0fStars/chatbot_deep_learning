import random
import json
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize, get_training_device

from constants import *

device = get_training_device()
with open('intents.json', 'r') as file:
    intents = json.load(file)

    torch_file_path = 'data.pth'
    data = torch.load(torch_file_path)
    
    input_size = data[INPUT_SIZE]
    hidden_size = data[HIDDEN_SIZE]
    output_size = data[OUTPUT_SIZE]
    all_words = data[ALL_WORDS]
    tags = data[TAGS]
    model_state = data[MODEL_STATE]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # load the learned parameters
    model.load_state_dict(model_state)

    # set model to evaluation mode
    model.eval()

    bot_name = 'Alan'
    print("Lets chat! type 'quit' to exit")

    while True:
        sentence = input('You: ')

        if sentence == 'quit':
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        # 1 row because we have one sample
        # X.shape[0] number of columns
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]        
        
        if prob.item() > 0.75:
            for intent in intents[INTENTS]:
                if tag == intent[TAG]:
                    print(f'{bot_name}: {random.choice(intent[RESPONSES])}')
        else:
            print(f'{bot_name}: I do not understand...')