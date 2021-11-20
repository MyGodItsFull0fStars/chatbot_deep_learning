
from typing import Tuple

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from constants import device

# used as source: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354

def get_true_positives(y_true: torch.Tensor, y_predictions: torch.Tensor) -> torch.float32:
    return (y_true * y_predictions).sum().to(torch.float32)


def get_false_positives(y_true: torch.Tensor, y_predictions: torch.Tensor) -> torch.float32:
    return ((1 - y_true) * y_predictions).sum().to(torch.float32)


def get_true_negatives(y_true: torch.Tensor, y_predictions: torch.Tensor) -> torch.float32:
    return ((1 - y_true) * (1 - y_predictions)).sum().to(torch.float32)


def get_false_negatives(y_true: torch.Tensor, y_predictions: torch.Tensor) -> torch.float32:
    return (y_true * (1 - y_predictions)).sum().to(torch.float32)


def get_precision(data_loader: DataLoader, model: Module) -> Tuple[float, float, float]:
    """This function returns a tuple with the precision of the model and
    the true positives and total predicted positives

    Precision = True Positives / Total Predicted Positives
    """
    if data_loader is None or model is None:
        raise ValueError('invalid parameters')

    with torch.no_grad():
        true_positives: torch.Tensor = None
        false_positives: torch.Tensor = None
        precision: float = None

        for words, labels in data_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            _, predicted = torch.max(outputs.data, 1)

            true_positives = get_true_positives(labels, predicted)
            false_positives = get_false_positives(labels, predicted)
            total_predicted_positives = true_positives + false_positives

            precision = true_positives / total_predicted_positives

    return precision, true_positives, total_predicted_positives


def get_recall(data_loader: DataLoader, model: Module) -> Tuple[float, float, float]:
    if data_loader is None or model is None:
        raise ValueError('invalid parameters')

    with torch.no_grad():
        true_positives: torch.Tensor = None
        false_negative: torch.Tensor = None
        recall: float = None

        for words, labels in data_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            _, predicted = torch.max(outputs.data, 1)

            true_positives = get_true_positives(labels, predicted)
            false_negative = get_false_negatives(labels, predicted)
            denominator = true_positives + false_negative

            recall = true_positives / denominator

    return recall, true_positives, denominator


def get_accuracy(data_loader: DataLoader, model: Module) -> Tuple[float, int, int]:
    """This function returns a tuple with the accuracy of the model and the correct and total labels

    Accuracy = Number of correct predictions / Total number of predictions
    """
    # disables gradient calculations
    if data_loader is None or model is None:
        raise ValueError('invalid parameters')

    with torch.no_grad():
        correct = 0
        total = 0

        for words, labels in data_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (correct/total, correct, total)


class PerformanceMeasures():

    def __init__(self, data_loader: DataLoader, model: Module) -> None:
        if data_loader is None or model is None:
            self.accuracy: torch.float32 = 0
            self.precision: torch.float32 = 0
            self.recall: torch.float32 = 0
            self.f1_score: torch.float32 = 0
        else:
            self.accuracy, self.precision, self.recall, self.f1_score = PerformanceMeasures.__init_member_values(
                data_loader, model)

    @staticmethod
    def __init_member_values(data_loader: DataLoader, model: Module) -> Tuple[torch.float32, torch.float32, torch.float32, torch.float32]:
        epsilon: torch.float32 = 1e-7

        with torch.no_grad():
            true_positives: torch.float32 = 0
            false_positives: torch.float32 = 0
            true_negatives: torch.float32 = 0
            false_negatives: torch.float32 = 0
            counter: torch.int32 = 0
            total: torch.int32 = 0

            for words, labels in data_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                outputs = model(words)

                _, predicted = torch.max(outputs.data, 1)

                true_positives += get_true_positives(labels, predicted)
                false_positives += get_false_positives(labels, predicted)
                true_negatives += get_true_negatives(labels, predicted)
                false_negatives += get_false_negatives(labels, predicted)
                total += labels.size(0)
                # counter used to calculate value/N
                counter += 1

            true_positives /= counter
            false_positives /= counter
            true_negatives /= counter
            false_negatives /= counter

            accuracy = (true_positives + true_negatives) / (total)
            precision = true_positives / \
                (true_positives + false_positives + epsilon)
            recall = true_positives / \
                (true_positives + false_negatives + epsilon)
            f1_score = 2 * (precision * recall) / \
                (precision + recall + epsilon)

            return accuracy, precision, recall, f1_score
