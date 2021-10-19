import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(NeuralNet, self).__init__()
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
