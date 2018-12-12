from torch import nn

from modules.dropout import RNNDropout


class Gate(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(input_size, input_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return inputs * self.gate(inputs)
