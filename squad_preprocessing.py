from squad import Squad, Squad_Transform
from utils import *

SQUAD_FILE_PATH: str = 'squad_dataset.json'
ignore_words = ['?', '!', '.', ',', ';',
                '!', '#', '$', '%', '-', '--', "'",
                '&', '(', ')', '⟨', '⟩', '`']


def is_valid_word(word: str) -> bool:
    # don't accept empty words or single characters
    if len(word) <= 1:
        return False

    # if whole word or beginning of word is a number, word is not valid
    if word.isdigit() or word[0].isdigit():
        return False

    # don't accept word if a invalid word is contained in it
    for iw in ignore_words:
        if iw in word:
            return False
    return True

all_words = []
tags = []

squad = Squad(SQUAD_FILE_PATH)
squad_transform = Squad_Transform(squad)

for qas_title, qas_list in squad_transform.title_question_answer_dict.items():
    tags.append(qas_title)

    for qas_element in qas_list:
        question = tokenize(qas_element.question)
        all_words.extend(question)

all_words = [stemming(
    word) for word in all_words if is_valid_word(word)]

all_words = get_sorted_unique_string_list(all_words)
tags = get_sorted_unique_string_list(tags)

with open('all_words.txt', 'w') as all_words_file:
    all_words_file.write(' '.join(word for word in all_words))

with open('tags.txt', 'w') as tags_file:
    tags_file.write(' '.join(word for word in tags))
