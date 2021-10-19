import json

from squad import *

SQUAD_FILE_PATH: str = 'squad_dataset.json'

with open(SQUAD_FILE_PATH, 'r') as squad_file:
    ignore_words = ['?', '!', '.', ',', ';', '!', '#', '$', '%', '-', '--']
    all_words = []
    X_y = []
    tags = []


    squad_json = json.load(squad_file)

    squad = Squad(squad_json)

    for squad_data in squad.data_list:
        tags.append(squad_data.title)

        for paragraph in squad_data.paragraphs:
            # TODO maybe insert again for better results
            # self.tags.append(paragraph.context)

            for question_answer_set in paragraph.question_answer_sets:
                question = tokenize(question_answer_set.question)
                all_words.extend(question)

    all_words = [stemming(word) for word in all_words if word not in ignore_words]

    all_words = get_sorted_unique_string_list(all_words)
    tags = get_sorted_unique_string_list(tags)

    with open('all_words.txt', 'w') as all_words_file:
        all_words_file.write(' '.join(word for word in all_words))

    with open('tags.txt', 'w') as tags_file:
        tags_file.write(' '.join(word for word in tags))


