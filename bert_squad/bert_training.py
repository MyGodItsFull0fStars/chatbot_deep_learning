# %% [markdown]
# # BERT Training
# 
# This Jupyter notebook contains the training of a BERT model and the necessary steps to convert the data to be usable for the training.
# 
# Author: Chrisitan Bauer

# %% [markdown]
# ## Sources:
# 
# - https://towardsdatascience.com/how-to-train-bert-for-q-a-in-any-language-63b62c780014
# - https://huggingface.co/transformers/v3.4.0/model_doc/bert.html#bertforquestionanswering
# - https://huggingface.co/docs/datasets/metrics.html
# 

# %% [markdown]
# ## WandB Montitor Output
# 
# https://wandb.ai/my-god-its-full-of-stars/BERT%20Training?workspace=user-my-god-its-full-of-stars

# %%
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertForQuestionAnswering, BertTokenizerFast, 
    AdamW, 
    DistilBertForQuestionAnswering, DistilBertTokenizer)

# fix BERT to DistillBERT transition
# https://fixexception.com/transformers/char-to-token-is-not-available-when-using-python-based-tokenizers/
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding as EncodingFast
    
from datasets import load_dataset, load_metric
import wandb
from tqdm.notebook import tqdm, trange, tqdm_notebook # show progress bar
from sklearn.metrics import f1_score

# %% [markdown]
# ## Load Dataset and Split into Training and Validation Datasets

# %%
dataset = load_dataset("squad")
# metric = load_metric('bertscore')
train_dataset, validation_dataset = dataset['train'], dataset['validation']
# train_dataset

# %% [markdown]
# ### Question-Answer Set Structure
# 
# Each Set conists of the following compontens:
# 
# - **Question**: string contianing the question Bert gets asked.
# - **Context**: a larger sequence containing the answer to the question.
# - **Answer**: a part of the context, that answers the question.
# 
# Given the question and context, the model must be able to read both of them and return the token positions of the predicted answer within the context.

# %%
# train_dataset[0]

# %%
# answer_start = train_dataset[0]['answers']['answer_start'][0]
# answer_start

# %%
# answer_length = len(train_dataset[0]['answers']['text'][0])
# answer_length

# %%
# train_dataset[0]['context'][answer_start:answer_start+answer_length]

# %% [markdown]
# ## Formatting QA-Sets
# 
# Before the training can start, the answer section of the sets needs to be reformatted.
# The key `answer_end` gets added to the dictionary and for easier accessing, the lists containing only one element get removed by the element itself.

# %%
def add_end_idx(answers, contexts):
    new_answers = []
    # loop through each answer-context pair
    for answer, context in tqdm(zip(answers, contexts)):
        # quick reformating to remove lists
        answer['text'] = answer['text'][0]
        answer['answer_start'] = answer['answer_start'][0]
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
        new_answers.append(answer)
    return new_answers

def prep_data(dataset):
    questions = dataset['question']
    contexts = dataset['context']
    answers = add_end_idx(
        dataset['answers'],
        contexts
    )
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }

# %% [markdown]
# ## Initialize Model, Tokenizer and Optimizer

# %%
train_dataset = prep_data(train_dataset)
validation_dataset = prep_data(validation_dataset)
train_dataset['answers'][:3]

# %% [markdown]
# ### Tokenization
# 
# For Bert to be able to read the SQUAD dataset, tokenization is required.
# For `context` and `question` the standard `tokenizer()` method can be used.
# 
# This method encodes both `context` and `question` strings into single arrays of tokens.
# This will act as the input for the QA training.

# %%
pretrained_model_name: str = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
# tokenize
# train = tokenizer(train_dataset['context'], train_dataset['question'],
#                           truncation=True, padding='max_length',
#                           max_length=512, return_tensors='pt')

# validation = tokenizer(validation_dataset['context'], validation_dataset['question'],
#                           truncation=True, padding='max_length',
#                           max_length=512, return_tensors='pt')
            

# %%
# tokenizer.decode(train['input_ids'][0])[:855]
# tokenizer.decode(validation['input_ids'][0])[:855]

# %% [markdown]
# ### Tokenization Part 2
# 
# Since tokens get fed into Bert, the start and end positions of the tokens need to be provided.
# 
# This is done by the method `add_token_positions()`.

# %%
def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in tqdm(range(len(answers))):
        # append start/end token position using char_to_token method
        # encodings = BatchEncoding([answers[i]['answer_start'], answers[i]['answer_end']], encoding=EncodingFast())
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            # encodings = BatchEncoding(answers[i]['answers_end'] -shift, encoding=EncodingFast())
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


# %%
# apply function to the data
# add_token_positions(train, train_dataset['answers'])
# add_token_positions(validation, validation_dataset['answers'])

# %% [markdown]
# ### This function adds two more tensors to the `Encoding` object (which will be fed into Bert)
# 
# - the `start_positions` and
# - the `end_positions` tensors.
# 

# %%
# train.keys()

# %%
# train['start_positions'][:5], train['end_positions'][:5]

# %% [markdown]
# ### Training
# 
# For the training, Pytorch is used and the dataset will be converted to a Pytorch Dataset.

# %%
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# train_dataset = SquadDataset(train)
# validation_dataset = SquadDataset(validation)

# %%
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# len(train_loader)

# %%
# validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
# len(validation_loader)

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    threshold=0.001, 
    cooldown=0, 
    eps=1e-07, 
    patience=2, 
    verbose=True, 
    min_lr=1e-5, 
    factor=0.5)

# %% [markdown]
# ### Helper Function for partially loading the dataset
# 
# This is required because otherwise an error occurs because to much RAM is allocated.

# %%
def get_next_dataloader(start_index: int, end_index: int, shuffle: bool = True, dataset_key: str = 'train') -> DataLoader:
    part_dataset = dataset[dataset_key][start_index:end_index]
    part_dataset = prep_data(part_dataset)
    part_train = tokenizer(part_dataset['context'], part_dataset['question'],
                          truncation=True, padding='max_length',
                          max_length=512, return_tensors='pt')
    tokenizer.decode(part_train['input_ids'][0])[:855]
    # apply function to the data
    add_token_positions(part_train, part_dataset['answers'])
    part_dataset = SquadDataset(part_train)
    return DataLoader(part_dataset, batch_size=32, shuffle=shuffle)




# %%
wandb.init(
    project='BERT Training',
    config={
        'batch_size': 32,
        'dataset': 'SQUAD 2.0'
    })

def convert_all_input_ids_to_tokens(input_ids):
    return [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]



# %%
max_train_batch_length: int = 2738
max_validation_batch_length: int = 331

num_episodes: int = 100
save_zones = [epoch for epoch in range(10, 101, 25)]
step = 5

training_losses = []
validation_losses = []

def train_model(start_epoch: int, end_epoch: int, model, optimizer, scheduler):
    
    # for epoch in trange(start_epoch, end_epoch):
    for epoch in range(start_epoch, end_epoch):
        print(f'Epoch: {epoch}')

        model.train()
        train_accuracy = 0
        train_total = 0

        f1_start = 0
        f1_end = 0

    # roberta-small
        # for start_range in trange(0, max_train_batch_length, step):
        for start_range in range(0, max_train_batch_length, step):
            end_range = min(max_train_batch_length, start_range + step)

            # loop = tqdm_notebook(get_next_dataloader(start_range, end_range), desc=f'<{start_range}:{end_range}>')
            # loop.set_description(f'Epoch {epoch + 1} Training <{start_range}:{end_range}>')
            loop = get_next_dataloader(start_range, end_range)

            for batch in loop:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask,
                                start_positions=start_positions,
                                end_positions=end_positions)
                
    
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits
                answer_start = torch.argmax(start_scores, dim=1)
                answer_end = torch.argmax(end_scores, dim=1)

                train_accuracy += (start_positions == answer_start).sum().float()
                train_accuracy += (end_positions == answer_end).sum().float()

                # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
                f1_start += f1_score(start_positions.cpu(), answer_start.cpu(), average='weighted', zero_division=0)
                f1_end += f1_score(end_positions.cpu(), answer_end.cpu(), average='weighted', zero_division=0)


                train_total += len(start_positions)
            
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # loop.set_postfix(loss=loss.item())


        training_losses.append(loss.item())
        wandb.log({
            'training loss': loss.item(),
            'training accuracy': train_accuracy / (train_total * 2),
            'training f1 score average': (f1_start + f1_end) / (train_total * 2),
            'training f1 score start position': f1_start / train_total,
            'training f1 score end position': f1_end / train_total
        })
        

        validation_accuracy = 0
        validation_total = 0

        f1_start = 0
        f1_end = 0

        # for start_range in trange(0, max_validation_batch_length, step):
        for start_range in range(0, max_validation_batch_length, step):
            end_range = min(max_validation_batch_length, start_range + step)
            # loop = tqdm(get_next_dataloader(start_range, end_range, shuffle=False, dataset_key='validation'))
            loop = get_next_dataloader(start_range, end_range, shuffle=False, dataset_key='validation')
            # loop.set_description(f'Epoch {epoch + 1} Validation <{start_range}:{end_range}>')
            

            with torch.no_grad():
                model.eval()
                for batch in loop:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)

                    start_scores = outputs.start_logits
                    end_scores = outputs.end_logits
                    answer_start = torch.argmax(start_scores, dim=1)
                    answer_end = torch.argmax(end_scores, dim=1)

                    validation_accuracy += (start_positions == answer_start).sum().float()
                    validation_accuracy += (end_positions == answer_end).sum().float()

       
                    f1_start += f1_score(start_positions.cpu(), answer_start.cpu(), average='weighted', zero_division=0)
                    f1_end += f1_score(end_positions.cpu(), answer_end.cpu(), average='weighted', zero_division=0)

                    validation_total += len(start_positions)
                    
                    loss = outputs.loss

                    # loop.set_postfix(loss=loss.item())

                    scheduler.step(loss)

        validation_losses.append(loss.item())
        wandb.log({
            'validation loss': loss.item(), 
            'validation accuracy': validation_accuracy / (validation_total * 2),
            'training f1 score average': (f1_start + f1_end) / (train_total * 2),
            'training f1 score start position': f1_start / train_total,
            'training f1 score end position': f1_end / train_total
        })

    torch.save(model.state_dict(), f'models/bert_model_{epoch}.pt')
    torch.save(optimizer.state_dict(), f'models/optimizer_{epoch}.pt')
    torch.save(scheduler.state_dict(), f'models/scheduler_{epoch}.pt')



def init_model(epoch: int = 0):
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        threshold=0.001, 
        cooldown=0, 
        eps=1e-07, 
        patience=2, 
        verbose=True, 
        min_lr=1e-5, 
        factor=0.5)

    if epoch != -1:
        model_dict = torch.load(f'models/bert_model_{epoch}.pt')
        model.load_state_dict(model_dict)
        optimizer_dict = torch.load(f'models/optimizer_{epoch}.pt')
        optimizer.load_state_dict(optimizer_dict)
        scheduler_dict = torch.load(f'models/scheduler_{epoch}.pt')
        scheduler.load_state_dict(scheduler_dict)

    return model, optimizer, scheduler

        
def get_last_saved_epoch(path: str = 'models/') -> int:
    from os import walk
    files = []
    for (_, _, filenames) in walk(path):
        filenames = [file for file in filenames if file.endswith('.pt')]
        files.extend(filenames)

    epoch_list = [int(file.split('.')[0].split('_').pop()) for file in files]

    return max(epoch_list) if len(epoch_list) > 0 else -1


saved_epoch = get_last_saved_epoch()
step: int = 1
for start_index in range(saved_epoch, saved_epoch + 10, step):
    model, optimizer, scheduler = init_model(saved_epoch)
    start_index += 1
    train_model(start_index, start_index + step, model, optimizer, scheduler)
    del model
    del optimizer
    del scheduler
    



