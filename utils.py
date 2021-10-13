# source: https://www.youtube.com/watch?v=8qwowmiXANQ
from typing import List
import nltk
import numpy as np
import torch
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence: str) -> List[str]:
    return nltk.word_tokenize(sentence)

def stemming(word: str) -> str:
    return stemmer.stem(word.lower())

def get_sorted_unique_string_list(word_list: List[str]) -> List[str]:
    return sorted(set(word_list))

def bag_of_words(tokenized_sentence, all_words) -> List[str]:
    tokenized_sentence = [stemming(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1

    return bag

def get_training_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# words = ['Organize', 'organizes', 'organization', 'organizing']
# stemmed_words = [stemming(word) for word in words]
# print(stemmed_words)