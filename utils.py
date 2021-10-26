# source: https://www.youtube.com/watch?v=8qwowmiXANQ
from typing import Dict, List
import nltk
import numpy as np
import torch
from functools import reduce

# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence: str) -> List[str]:
    return nltk.word_tokenize(sentence)


def stemming(word: str) -> str:
    return stemmer.stem(word.lower())


def get_sorted_unique_string_list(word_list: List[str]) -> List[str]:
    return sorted(set(word_list))


def bag_of_words(tokenized_sentence, all_words_dict: dict) -> List[str]:
    tokenized_sentence = [stemming(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words_dict.keys()), dtype=np.float32)

    for word in tokenized_sentence:
        if word in all_words_dict:
            idx = all_words_dict[word]
            bag[idx] = 1

    return bag

def get_average(data: list) -> float:
    return reduce(lambda a, b: a + b, data) / len(data)


def get_training_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_all_words_dict(all_words) -> Dict[str, int]:
    all_words_dict = {}

    for idx, word in enumerate(all_words):
        all_words_dict[word] = idx

    return all_words_dict

def get_all_words() -> List[str]:
    with open('all_words.txt', 'r') as word_file:
        word_str = word_file.read()
        all_words = word_str.split(' ')

        return all_words

def get_tags() -> List[str]:
    with open('tags.txt', 'r') as tags_file:
        tags_str = tags_file.read()
        tags = tags_str.split(' ')

        return tags