import torch.nn as nn

class NeuralNetSmall(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNetSmall, self).__init__()
        # nn.Linear(input_size, output_size)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax because of CrossEntropyLoss!
        return out


class NeuralNetMedium(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNetMedium, self).__init__()
        # nn.Linear(input_size, output_size)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 2)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        # no activation and no softmax because of CrossEntropyLoss!
        return out

class NeuralNetGrande(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNetGrande, self).__init__()
        # nn.Linear(input_size, output_size)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size * 4)
        self.l3 = nn.Linear(hidden_size * 4, hidden_size * 8)
        self.l4 = nn.Linear(hidden_size * 8, hidden_size * 4)
        self.l5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.l6 = nn.Linear(hidden_size * 2, hidden_size)
        self.l7 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        # no activation and no softmax because of CrossEntropyLoss!
        return out
