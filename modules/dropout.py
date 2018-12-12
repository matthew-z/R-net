from torch import nn


class RNNDropout(nn.Module):
    def __init__(self, p, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, inputs):

        if not self.training:
            return inputs
        if self.batch_first:
            mask = inputs.new_ones(inputs.size(0), 1, inputs.size(2), requires_grad=False)
        else:
            mask = inputs.new_ones(1, inputs.size(1), inputs.size(2), requires_grad=False)
        return self.dropout(mask) * inputs
