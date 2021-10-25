import json
from typing import Any, Dict, List, Tuple
from constants import MODEL_NAME, ACCURACY_KEY, LOSS_KEY
from os import path, mkdir, rmdir


def init_accuracy_loss_json_file(model_name: str, dir_name: str, file_name: str) -> bool:
    dictionary = _create_json_dictionary(model_name)
    file_path = _prepare_json_file_path(dir_name, file_name)

    if not path.exists(file_path):
        with open(file_path, 'w') as out_file:
            json.dump(dictionary, out_file)
            return True

    # File does already exist
    return False


def get_values_from_json(dir_name: str, file_name: str) -> Tuple[str, List[float], List[float]]:
    file_path = _prepare_json_file_path(dir_name, file_name)
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        if MODEL_NAME in data and ACCURACY_KEY in data and LOSS_KEY in data:
            return data[MODEL_NAME], data[ACCURACY_KEY], data[LOSS_KEY]
        else:
            raise ValueError(
                'The loaded json file is not in the required form')


def update_accuracy_loss_json_file(dir_name: str, file_name: str, accuracy: List[float], loss: List[float]):
    model_name, json_accuracy, json_loss = get_values_from_json(
        dir_name, file_name)
    # insert new values after the last position of the pre-existing json values
    json_accuracy.extend(accuracy)
    json_loss.extend(loss)

    new_json_dict = _create_json_dictionary(
        model_name, json_accuracy, json_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def update_accuracy(dir_name: str, file_name: str, accuracy: List[float]):
    model_name, json_file_accuracy, json_file_loss = get_values_from_json(
        dir_name, file_name)
    json_file_accuracy.extend(accuracy)

    new_json_dict = _create_json_dictionary(
        model_name, json_file_accuracy, json_file_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def update_loss(dir_name: str, file_name: str, loss: List[float]):
    model_name, json_file_accuracy, json_file_loss = get_values_from_json(
        dir_name, file_name)
    json_file_loss.extend(loss)

    new_json_dict = _create_json_dictionary(
        model_name, json_file_accuracy, json_file_loss)

    _write_json_dict_to_file(dir_name, file_name, new_json_dict)


def _create_json_dictionary(model_name: str, accuracy: List[float] = None, loss: List[float] = None) -> Dict[str, Any]:
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

    file_path: str = file_name if dir_name is None else f'{dir_name}/{file_name}'

    if not file_path.endswith('.json'):
        file_path = f'{file_path}.json'

    return file_path


def _make_dir_if_not_exist(dir_name: str) -> str:
    if dir_name is not None and not path.exists(dir_name):
        mkdir(dir_name)
