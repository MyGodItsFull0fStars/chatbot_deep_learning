# source of dataset: https://rajpurkar.github.io/SQuAD-explorer/

import json
from typing import Any, Dict, List, Tuple
from copy import copy

from torch.utils import data

VERSION_KEY: str = 'version'
DATA_KEY: str = 'data'

TITLE_KEY: str = 'title'
PARAGRAPHS_KEY: str = 'paragraphs'

CONTEXT_KEY: str = 'context'
QAS_KEY: str = 'qas'

class Question_Answer_Set(object):
    
    def __init__(self, qas_entry) -> None:

        self.question = qas_entry['question']
        self.is_impossible = qas_entry['is_impossible']
        # if it is impossible to verify, the answers are considered as plausible answers
        self.answers = qas_entry['plausible_answers' if self.is_impossible == True else 'answers']

class Paragraphs():

    def __init__(self, paragraph_entry) -> None:
        self.context = paragraph_entry[CONTEXT_KEY]
        self.question_answer_sets: List[Question_Answer_Set] = [Question_Answer_Set(data) for data in paragraph_entry['qas']]


class Squad_Data():

    def __init__(self, entry) -> None:
        self.title: str = entry[TITLE_KEY]
        self.paragraphs: List[Paragraphs] = [Paragraphs(data) for data in entry[PARAGRAPHS_KEY]]


class Squad():
    # The Stanford Question Answering Dataset 2.0
    def __init__(self, data_list: list) -> None:
        self.data_list: List[Squad_Data] = [Squad_Data(data) for data in data_list]

with open('train-v2.0.json', 'r') as file:
    training_questions = json.load(file)
    
    # x = json.loads(json.dumps(training_questions), object_hook=lambda d: SimpleNamespace(**d))

    data = training_questions['data']
    squad = Squad(data)

    for para in squad.data_list[0].paragraphs:
        print(para.context)
 