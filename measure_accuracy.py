import os
import threading
from queue import Queue
from typing import List, Tuple

from torch.utils.data.dataloader import DataLoader

import json_utils
import model_utils

lock = threading.Lock()
queue = Queue()

directory_name: str = 'models_1.5_hl'

accuracy_list: List[Tuple[str, float]] = []

from_range: int = 0
to_range: int = 44
batch_size: int = 128
num_workers: int = 10
train_loader: DataLoader = (from_range, to_range, batch_size, num_workers)


def threader():
    while True:
        model_name = queue.get()
        # do stuff
        __calculate_accuracy(directory_name, model_name)
        queue.task_done()


def get_model_list_file_names(dir_name: str) -> List[str]:
    model_files: List[str] = []

    if os.path.exists(dir_name):
        os.chdir(dir_name)
        all_files: List[str] = os.listdir()
        model_files = [
            file_name for file_name in all_files if file_name.endswith('.pth')]
        os.chdir('..')

    return sorted(model_files)


def __calculate_accuracy(dir_name: str, model_file_name: str) -> None:

    model_name: str = model_file_name.removesuffix('.pth')
    print(f'calculate accuracy for model: {model_name}')
    model = model_utils.get_model(dir_name, model_file_name)
    accuracy, _, _ = model_utils.get_accuracy(train_loader, model)

    lock.acquire()
    accuracy_list.append((model_name, accuracy))
    lock.release()


def __init_threads(num_of_threads: int, model_list: List[str]):
    # starting a number of threads for the operations
    for _ in range(num_of_threads):
        thread = threading.Thread(target=threader)
        thread.daemon = True
        thread.start()

    for model_name in model_list:
        queue.put(model_name)

    queue.join()


def main():

    model_list = get_model_list_file_names(directory_name)
    max_thread_number: int = 5

    if len(model_list) == 0:
        raise ValueError('Model list is empty')

    num_of_threads: int = max_thread_number if len(
        model_list) > max_thread_number else len(model_list)

    __init_threads(num_of_threads, model_list)

    json_utils.update_accuracy(
        directory_name, json_utils.DEFAULT_FILE_NAME, accuracy_list)

    print('done')


if __name__ == '__main__':
    main()
