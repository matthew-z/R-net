"""Embedding each word using word-level and char-level embedding"""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from modules import RNN


class CharLevelEmbedding(nn.Module):
    def __init__(self, vocab_size, char_embedding_tensor=None, char_embedding_dim=300, output_dim=300,
                 padding_idx=None, bidirectional=True, cell_type="gru", num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=char_embedding_dim, padding_idx=padding_idx)
        if char_embedding_tensor is not None:
            self.embedding.weight.data.copy_(char_embedding_tensor)
        self.network = RNN(char_embedding_dim, output_dim, bidirectional=bidirectional,
                           cell_type=cell_type, num_layers=num_layers, pack=True, batch_first=True)

        if bidirectional:
            self.projection_layer = nn.Linear(output_dim * 2, output_dim)

    def forward(self, words_tensor, lengths):
        """

        :param words_tensor: tuple of (words_tensor (B x T), lengths)
        :return:
        """
        embed = self.embedding(words_tensor)
        embed_pack = pack_padded_sequence(embed, lengths, batch_first=True)
        outputs, hidden = self.network(embed_pack)

        if hasattr(self, "projection_layer"):
            batch_size = len(words_tensor)
            hidden = self.projection_layer(hidden.view(batch_size, -1))

        return hidden