# source of dataset: https://rajpurkar.github.io/SQuAD-explorer/

import json
from typing import Any, Dict, List, Tuple
from copy import deepcopy

from torch.utils import data
from utils import *

QUESTION_KEY: str = 'question'
ANSWERS_KEY: str = 'answers'
PLAUSIBLE_ANSWERS_KEY: str = 'plausible_answers'

class Question_Answer_Set(object):
    
    def __init__(self, qas_entry) -> None:

        self.question = qas_entry['question']
        self.is_impossible = qas_entry['is_impossible']
        # if it is impossible to verify, the answers are considered as plausible answers
        self.answers = qas_entry[PLAUSIBLE_ANSWERS_KEY if self.is_impossible == True else ANSWERS_KEY]

CONTEXT_KEY: str = 'context'
QUESTION_ANSWER_SET_KEY: str = 'qas'
class Paragraphs():

    def __init__(self, paragraph_entry) -> None:
        self.context = paragraph_entry[CONTEXT_KEY]
        self.question_answer_sets: List[Question_Answer_Set] = [Question_Answer_Set(data) for data in paragraph_entry[QUESTION_ANSWER_SET_KEY]]

    def __getitem__(self, index) -> Question_Answer_Set:
        qas = deepcopy(self.question_answer_sets[index])
        return qas


TITLE_KEY: str = 'title'
PARAGRAPHS_KEY: str = 'paragraphs'
class Squad_Data():

    def __init__(self, entry) -> None:
        self.title: str = entry[TITLE_KEY]
        self.paragraphs: List[Paragraphs] = [Paragraphs(data) for data in entry[PARAGRAPHS_KEY]]

    def __getitem__(self, index) -> Paragraphs:
        paragraph = deepcopy(self.paragraphs[index])
        return paragraph


VERSION_KEY: str = 'version'
DATA_KEY: str = 'data'

class Squad():
    # The Stanford Question Answering Dataset 2.0
    def __init__(self, json_file) -> None:
        self.version: str = json_file['version']
        self.data_list: List[Squad_Data] = [Squad_Data(data) for data in json_file['data']]

    def __getitem__(self, index) -> Squad_Data:
        squad_data = deepcopy(self.data_list[index])
        return squad_data

# with open('squad_dataset.json', 'r') as file:
#     training_questions = json.load(file)

#     squad = Squad(training_questions)

#     for para in squad[0].paragraphs[:1]:
#         # print(para.context)
#         word_list = [stemming(word) for word in para.context.split(' ')]
#         word_list = get_sorted_unique_string_list(word_list)
#         # word_list = get_sorted_unique_string_list(tokenize(para.context))
#         # word_list = [stemming(word) for word in word_list]
#         print(word_list)

 
