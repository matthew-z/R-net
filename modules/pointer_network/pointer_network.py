from allennlp.common import Registrable
from torch import nn
import torch
from allennlp.nn.util import masked_softmax
from modules.dropout import RNNDropout
from overrides import overrides

class QAOutputLayer(nn.Module, Registrable):
    pass

@QAOutputLayer.register("pointer_network")
class PointerNetwork(QAOutputLayer):
    def __init__(self, question_size, passage_size, attention_size=75,
                 cell_type=nn.GRUCell, dropout=0, batch_first=False):
        super().__init__()

        self.batch_first = batch_first

        # TODO: what is V_q? (section 3.4)
        v_q_size = question_size

        self.cell = cell_type(passage_size, question_size)
        self.dropout = dropout

        self.passage_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(question_size + passage_size, attention_size, bias=False),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

        self.question_linear = nn.Sequential(
            RNNDropout(dropout),
            nn.Linear(question_size + v_q_size, attention_size, bias=False),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

        self.V_q = nn.Parameter(torch.randn(1, 1, v_q_size), requires_grad=True)

    @overrides
    def forward(self, question, question_mask, passage, passage_mask):
        """
        :param question: T B H
        :param question_mask: T B
        :param passage:
        :param passage_mask:
        :return: B x 2
        """

        if self.batch_first:
            question = question.transpose(0, 1)
            question_mask = question_mask.transpose(0, 1)
            passage = passage.transpose(0, 1)
            passage_mask = passage_mask.transpose(0, 1)

        state = self._question_pooling(question, question_mask)
        cell_input, ans_start_logits = self._passage_attention(passage, passage_mask, state)
        state = self.cell(cell_input, hx=state)
        _, ans_end_logits = self._passage_attention(passage, passage_mask, state)


        return ans_start_logits.transpose(0, 1), ans_end_logits.transpose(0, 1)

    def _question_pooling(self, question, question_mask):
        V_q = self.V_q.expand(question.size(0), question.size(1), -1)
        logits = self.question_linear(torch.cat([question, V_q], dim=-1)).squeeze(-1)
        score = masked_softmax(logits, question_mask, dim=0)
        state = torch.sum(score.unsqueeze(-1) * question, dim=0)
        return state

    def _passage_attention(self, passage, passage_mask, state):
        state_expand = state.unsqueeze(0).expand(passage.size(0), -1, -1)
        logits = self.passage_linear(torch.cat([passage, state_expand], dim=-1)).squeeze(-1)
        score = masked_softmax(logits, passage_mask, dim=0)
        cell_input = torch.sum(score.unsqueeze(-1) * passage, dim=0)
        return cell_input, logits
