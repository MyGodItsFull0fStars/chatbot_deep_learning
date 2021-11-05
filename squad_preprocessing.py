import json

from squad import *

SQUAD_FILE_PATH: str = 'squad_dataset.json'
ignore_words = ['?', '!', '.', ',', ';',
                '!', '#', '$', '%', '-', '--', "'",
                '&', '(', ')']


def is_valid_word(word: str) -> bool:
    if len(word) <= 1:
        return False

    if word.isdigit():
        return False

    for iw in ignore_words:
        if iw in word:
            return False
    return True


with open(SQUAD_FILE_PATH, 'r') as squad_file:
    all_words = []
    X_y = []
    tags = []

    squad_json = json.load(squad_file)

    squad = Squad(squad_json)
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
