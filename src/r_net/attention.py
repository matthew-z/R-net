import torch
from torch import nn
from torch.nn import functional as F


class AttentionEncoding(nn.Module):
    def __init__(self):
        self.batch_first = False
        super().__init__()

    def _calculate_context(self, source, source_mask, score_unnormalized):
        """

        :return: if batch_first: seq x batch x n   context vectors
        """
        if source_mask is not None:
            score_unnormalized.data.masked_fill_(source_mask.data != 1, -1e12)
        n_batch, question_len, _ = source.size()
        scores = (F.softmax(score_unnormalized.view(-1, question_len))
                  .view(n_batch, question_len, 1))


        return scores

    def _pointer_output(self, source, source_mask, score_unnormalized):
        if source_mask is not None:
            score_unnormalized.data.masked_fill_(source_mask.data != 1, -1e12)
        n_batch, question_len, _ = source.size()
        output = score_unnormalized.view(n_batch, question_len)
        return output



class PairAttentionLayer(AttentionEncoding):
    def __init__(self, key_size, query1_size, query2_size,
                 attn_size=75, batch_first=False):
        super().__init__()
        self.key_linear = nn.Linear(key_size, attn_size, bias=False)
        self.query1_linear = nn.Linear(query1_size, attn_size, bias=False)
        self.query2_linear = nn.Linear(query2_size, attn_size, bias=False)
        self.score_linear = nn.Linear(attn_size, 1, bias=False)
        self.batch_first = batch_first

    def forward(self, keys, query1, query2, values=None, key_mask=None, return_key_scores=False):

        if not self.batch_first:
            keys = keys.transpose(0, 1)  # batch * seq_len * n
            query1 = query1.transpose(0, 1)  # batch * 1 * n
            query2 = query2.transpose(0, 1)  # batch * 1 * n
            key_mask = key_mask.transpose(0, 1)  # batch * seq_len * 1
        key_mask = key_mask.unsqueeze(2)

        curr_batch_size = query1.size(0)
        if keys.size(0) != curr_batch_size:
            keys = keys[:curr_batch_size]
            query2 = query2[:curr_batch_size]
            key_mask = key_mask[:curr_batch_size]

        if values is None:
            values = keys

        score_unnormalized = self.score_linear(F.tanh(self.key_linear(keys)
                                                      + self.query1_linear(query1)
                                                      + self.query2_linear(query2)))
        scores = self._calculate_context(keys, key_mask, score_unnormalized)
        # TODO: check dim order

        context = torch.bmm(scores.transpose(1, 2), values)
        if not self.batch_first:
            context = context.transpose(0, 1)

        if return_key_scores:
            return context, scores
        return context


class AttentionPooling(AttentionEncoding):
    def __init__(self, key_size, query_size, attn_size=75, batch_first=False):
        super().__init__()
        self.key_linear = nn.Linear(key_size, attn_size, bias=False)
        self.query_linear = nn.Linear(query_size, attn_size, bias=False)
        self.score_linear = nn.Linear(attn_size, 1, bias=False)
        self.batch_first = batch_first

    def forward(self, keys, query, key_mask=None, values=None, return_key_scores=False, broadcast_key=False):

        if not self.batch_first:
            keys = keys.transpose(0, 1)  # batch * seq_len * n
            query = query.transpose(0, 1)  # batch * 1 * n
            key_mask = key_mask.transpose(0, 1)  # batch * seq_len * 1

        if values is None:
            values = keys

        key_mask = key_mask.unsqueeze(2)

        curr_batch_size = query.size(0)
        if keys.size(0) != curr_batch_size and not broadcast_key:
            keys = keys[:curr_batch_size]
            key_mask = key_mask[:curr_batch_size]

        score_unnormalized = self.score_linear(F.tanh(self.key_linear(keys)
                                                      + self.query_linear(query)))

        scores = self._calculate_context(keys, key_mask, score_unnormalized)

        context = torch.bmm(scores.transpose(1, 2), values)
        if not self.batch_first:
            context = context.transpose(0, 1)
            scores = scores.transpose(0, 1)

        if return_key_scores:
            return context, scores
        return context
