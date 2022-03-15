import json
from os import mkdir, path
from typing import Any, Dict, List, Tuple

from constants import ACCURACY_KEY, LOSS_KEY, MODEL_NAME

DEFAULT_FILE_NAME: str = 'accuracy_loss_data.json'


def init_accuracy_loss_json_file(model_name: str, dir_name: str, file_name: str = DEFAULT_FILE_NAME) -> bool:
    dictionary = _create_json_dictionary(model_name)
    file_path = _prepare_json_file_path(dir_name, file_name)

    if not path.exists(file_path):
        with open(file_path, 'w') as out_file:
            json.dump(dictionary, out_file)
            return True

    # File does already exist
    return False


def get_values_from_json(dir_name: str, file_name: str = DEFAULT_FILE_NAME) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]]]:
    file_path = _prepare_json_file_path(dir_name, file_name)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        if MODEL_NAME in data and ACCURACY_KEY in data and LOSS_KEY in data:
            return data[MODEL_NAME], data[ACCURACY_KEY], data[LOSS_KEY]
        raise ValueError(
            'The loaded json file is not in the required form')


def update_accuracy_loss_json_file(dir_name: str, file_name: str, accuracy: List[Tuple[str, float]], loss: List[Tuple[str, float]]):
    model_name, json_accuracy, json_loss = get_values_from_json(
        dir_name, file_name)
    # insert new values after the last position of the pre-existing json values
    json_accuracy.extend(accuracy)
    json_loss.extend(loss)

    new_json_dict = _create_json_dictionary(
        model_name, json_accuracy, json_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def update_accuracy(dir_name: str, file_name: str = DEFAULT_FILE_NAME, accuracy: List[Tuple[str, float]] = None):
    model_name, json_file_accuracy, json_file_loss = get_values_from_json(
        dir_name, file_name)

    if accuracy is not None:
        json_file_accuracy.extend(accuracy)

    new_json_dict = _create_json_dictionary(
        model_name, json_file_accuracy, json_file_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def update_loss(dir_name: str = None, file_name: str = DEFAULT_FILE_NAME, loss: List[Tuple[str, float]] = []):
    model_name, json_file_accuracy, json_file_loss = get_values_from_json(
        dir_name, file_name)
    json_file_loss.extend(loss)

    new_json_dict = _create_json_dictionary(
        model_name, json_file_accuracy, json_file_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def _create_json_dictionary(model_name: str, accuracy: List[Tuple[str, float]] = None, loss: List[Tuple[str, float]] = None) -> Dict[str, Any]:
    dictionary = {
        MODEL_NAME: model_name,
        ACCURACY_KEY: accuracy if accuracy is not None else [],
        LOSS_KEY: loss if loss is not None else [],
    }
    return dictionary


def _write_json_dict_to_file(dir_name: str, file_name: str, json_dictionary: Dict[str, Any]):
    file_path = _prepare_json_file_path(dir_name, file_name)

    with open(file_path, 'w+') as out_file:
        out_file.write(json.dumps(json_dictionary))


def _prepare_json_file_path(dir_name: str, file_name: str) -> str:
    if file_name is None:
        raise ValueError('file_name has to be set')

    _make_dir_if_not_exist(dir_name)

    if dir_name is not None and dir_name.endswith('/'):
        dir_name = dir_name.removesuffix('/')

    file_path: str = file_name if dir_name is None else f'{dir_name}/{file_name}'

    if not file_path.endswith('.json'):
        file_path = f'{file_path}.json'

    return file_path


def _make_dir_if_not_exist(dir_name: str) -> str:
    if dir_name is not None and not path.exists(dir_name):
        mkdir(dir_name)


class AccuracyLossData():

    def __init__(self, dir_name: str) -> None:
        self.dir_name: str = dir_name
        self.file_path: str = self.__create_file_path(dir_name)
        self.model_name: str = None
        self.accuracies: Dict[str, float] = {}
        self.losses: Dict[str, float] = {}

        self.__init_json_member_variables()

    def __init_json_member_variables(self):
        with open(self.file_path, 'r') as json_file:
            json_data = json.load(json_file)
            self.model_name = json_data['model_name']
            self.__init_accuracies(json_data)
            self.__init_losses(json_data)

    def __init_accuracies(self, json_data):
        for accuracy in json_data['accuracy']:
            self.accuracies[accuracy[0]] = accuracy[1]

    def __init_losses(self, json_data):
        for loss in json_data['loss']:
            self.losses[loss[0]] = loss[1]

    def __create_file_path(self, dir_name: str, file_name: str = DEFAULT_FILE_NAME) -> str:
        if file_name is None:
            raise ValueError('file_name must not be None')
        if dir_name is None:
            return file_name
        if dir_name.endswith('/'):
            return f'{dir_name}{file_name}'
        return f'{dir_name}/{file_name}'
