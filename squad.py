# source of dataset: https://rajpurkar.github.io/SQuAD-explorer/

from typing import Any, Dict, List
from copy import deepcopy

from utils import *

QUESTION_KEY: str = 'question'
ANSWERS_KEY: str = 'answers'
PLAUSIBLE_ANSWERS_KEY: str = 'plausible_answers'


class Question_Answer_Set():

    def __init__(self, qas_entry) -> None:

        self.question: str = qas_entry['question']
        self.is_impossible: bool = qas_entry['is_impossible']
        # if it is impossible to verify, the answers are considered as plausible answers
        self.answer: str = self.__get_answer(qas_entry)

    @classmethod
    def from_single_params(cls, question: str, answer: str, is_impossible: bool):
        qas_entry: dict = {'question': question,
                           'answer': answer, 'is_impossible': is_impossible}
        return cls(qas_entry)

    def __get_answer(self, qas_entry) -> str:
        # get certain or probable answer
        answer = qas_entry[PLAUSIBLE_ANSWERS_KEY if self.is_impossible is True else ANSWERS_KEY]
        # answer consists of one list containing a dictionary with keys 'text' and 'answer_start'
        answer = answer.pop()
        # retrieve answer
        return answer['text']


CONTEXT_KEY: str = 'context'
QUESTION_ANSWER_SET_KEY: str = 'qas'


class Squad_Paragraph():

    def __init__(self, paragraph_entry) -> None:
        self.context = paragraph_entry[CONTEXT_KEY]
        self.question_answer_sets: List[Question_Answer_Set] = [
            Question_Answer_Set(data) for data in paragraph_entry[QUESTION_ANSWER_SET_KEY]]


TITLE_KEY: str = 'title'
PARAGRAPHS_KEY: str = 'paragraphs'


class Squad_Data():

    def __init__(self, entry) -> None:
        self.title: str = entry[TITLE_KEY]
        self.paragraphs: List[Squad_Paragraph] = [
            Squad_Paragraph(data) for data in entry[PARAGRAPHS_KEY]]

    def __getitem__(self, index) -> Squad_Paragraph:
        paragraph = deepcopy(self.paragraphs[index])
        return paragraph


VERSION_KEY: str = 'version'
DATA_KEY: str = 'data'


class Squad():
    # The Stanford Question Answering Dataset 2.0
    def __init__(self, json_file) -> None:
        self.version: str = json_file['version']
        self.data_list: List[Squad_Data] = [
            Squad_Data(data) for data in json_file['data']]

    def __getitem__(self, index) -> Squad_Data:
        squad_data = deepcopy(self.data_list[index])
        return squad_data


class Squad_Transform():

    def __init__(self, squad: Squad) -> None:
        self.title_question_answer_dict = self.transform_to_title_question_answer_structure(
            squad)

    def get_question_answer_set(self, question: str) -> List[Question_Answer_Set]:
        if question in self.title_question_answer_dict.keys():
            return self.title_question_answer_dict[question]
        else:
            return None

    def transform_to_title_question_answer_structure(self, squad: Squad) -> Dict[str, List[Question_Answer_Set]]:
        "Dict -> {title: Question Answer Set}"
        title_qas_dict: Dict[str, Question_Answer_Set] = {}

        for squad_paragraphs in squad:
            title = squad_paragraphs.title
            for paragraph in squad_paragraphs:
                title_qas_dict[title] = paragraph.question_answer_sets

        return title_qas_dict

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
