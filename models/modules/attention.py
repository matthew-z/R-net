import torch
from torch import nn
from torch.nn import functional as F


class AttentionPooling(nn.Module):
    def __init__(self, key_size, *query_sizes, attn_size=75, batch_first=False):
        super().__init__()
        self.key_linear = nn.Linear(key_size, attn_size, bias=False)
        self.query_linears = nn.ModuleList([nn.Linear(query_size, attn_size, bias=False) for query_size in query_sizes])
        self.score_linear = nn.Linear(attn_size, 1, bias=False)
        self.batch_first = batch_first

    def _calculate_scores(self, source, source_mask, score_unnormalized):
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

    def forward(self, key, queries, key_mask=None, values=None, return_key_scores=False, broadcast_key=False):
        """

        :param key:
        :param queries: a list of query
        :param key_mask:
        :param values:
        :param return_key_scores:
        :param broadcast_key:
        :return:
        """
        if not isinstance(queries, tuple) and not isinstance(queries, list):
            queries = (queries,)

        if not self.batch_first:
            key = key.transpose(0, 1)  # batch * seq_len * n
            queries = [query.transpose(0, 1) for query in queries]  # batch * 1 * n
            key_mask = key_mask.transpose(0, 1)  # batch * seq_len * 1


        key_mask = key_mask.unsqueeze(2)
        curr_batch_size = queries[0].size(0)

        if key.size(0) != curr_batch_size and not broadcast_key:
            key = key[:curr_batch_size]
            key_mask = key_mask[:curr_batch_size]

        if values is None:
            values = key

        score_before_transformation = self.key_linear(key)
        for i, query in enumerate(queries):
            score_before_transformation = score_before_transformation + self.query_linears[i](query)

        score_unnormalized = self.score_linear(F.tanh(score_before_transformation))
        scores = self._calculate_scores(key, key_mask, score_unnormalized)

        context = torch.bmm(scores.transpose(1, 2), values)
        if not self.batch_first:
            context = context.transpose(0, 1)
            score_unnormalized = score_unnormalized.transpose(0, 1)

        if return_key_scores:
            return context, score_unnormalized.squeeze(2)
        return context
