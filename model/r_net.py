import torch
from torch import nn
from model import module

class RNet(nn.Module):
    def __init__(self,
                 args,
                 char_embedding_config,
                 word_embedding_config,
                 sentence_encoding_config,
                 pair_encoding_config,
                 self_matching_config,
                 pointer_config):
        super().__init__()
        self.word_embedding = module.WordEmbedding(
            word_embedding=word_embedding_config["embedding_weights"],
            padding_idx=word_embedding_config["padding_idx"],
            requires_grad=word_embedding_config["update"])

        # self.char_embedding = module.CharLevelWordEmbeddingCNN(
        #     char_embedding_size=char_embedding_config["char_embedding_size"],
        #     char_num=char_embedding_config["char_num"],
        #     num_filters=char_embedding_config["num_filters"],
        #     ngram_filter_sizes=char_embedding_config["ngram_filter_sizes"],
        #     output_dim=char_embedding_config["output_dim"],
        #     activation=char_embedding_config["activation"],
        #     embedding_weights=char_embedding_config["embedding_weights"],
        #     padding_idx=char_embedding_config["padding_idx"],
        #     requires_grad=char_embedding_config["update"],
        # )

        self.r_net = _RNet(args,
                           self.word_embedding.output_dim, # + self.char_embedding.output_dim,
                           sentence_encoding_config,
                           pair_encoding_config,
                           self_matching_config,
                           pointer_config)

        self.args = args

    def forward(self,
                question,
                question_char,
                question_mask,
                passage,
                passage_char,
                passage_mask):
        # Embedding using Glove
        question, passage = self.word_embedding(question, passage)

        if torch.cuda.is_available():
            # question_char = question_char.cuda(self.args.device_id)
            # passage_char = passage_char.cuda(self.args.device_id)
            question = question.cuda(self.args.device_id)
            passage = passage.cuda(self.args.device_id)
            question_mask = question_mask.cuda(self.args.device_id)
            passage_mask = passage_mask.cuda(self.args.device_id)
        #
        # # char level embedding
        # question_char = self.char_embedding(question_char)
        # passage_char = self.char_embedding(passage_char)
        #
        # # concat word embedding and char level embedding
        # passage = torch.cat([passage, passage_char], dim=-1)
        # question = torch.cat([question, question_char], dim=-1)

        question = question.transpose(0, 1)
        question_mask = question_mask.transpose(0, 1)
        passage = passage.transpose(0, 1)
        passage_mask = passage_mask.transpose(0, 1)

        return self.r_net(question, question_mask, passage, passage_mask)

    def cuda(self, *args, **kwargs):
        "put word embedding on CPU"
        self.r_net.cuda(*args, **kwargs)
        # self.char_embedding.cuda(*args, **kwargs)
        return self


class _RNet(nn.Module):
    def __init__(self, args, input_size,
                 sentence_encoding_config,
                 pair_encoding_config,
                 self_matching_config, pointer_config):
        super().__init__()
        self.current_score = 0
        self.sentence_encoder = module.SentenceEncoding(
            input_size=input_size,
            hidden_size=sentence_encoding_config["hidden_size"],
            num_layers=sentence_encoding_config["num_layers"],
            bidirectional=sentence_encoding_config["bidirectional"],
            dropout=sentence_encoding_config["dropout"])

        sentence_encoding_direction = (2 if sentence_encoding_config["bidirectional"] else 1)
        sentence_encoding_size = (sentence_encoding_config["hidden_size"] * sentence_encoding_direction)

        self.pair_encoder = module.PairEncoderV2(
            memory_size=sentence_encoding_size,
            input_size=sentence_encoding_size,
            hidden_size=pair_encoding_config["hidden_size"],
            bidirectional=pair_encoding_config["bidirectional"],
            dropout=pair_encoding_config["dropout"],
            attention_size=pair_encoding_config["attention_size"]
        )

        pair_encoding_num_direction = (2 if pair_encoding_config["bidirectional"] else 1)
        pair_encoding_size = pair_encoding_config["hidden_size"] * pair_encoding_num_direction

        self.self_match_encoder = module.SelfMatchEncoder(
            memory_size=sentence_encoding_size,
            input_size=pair_encoding_size,
            hidden_size=self_matching_config["hidden_size"],
            dropout=self_matching_config["dropout"],
            attention_size=self_matching_config["attention_size"],
            bidirectional = self_matching_config["bidirectional"],
        )

        passage_size = pair_encoding_num_direction * pair_encoding_config["hidden_size"]

        self.pointer_net = module.OutputLayer(
            question_size=sentence_encoding_size,
            passage_size=passage_size,
            dropout=pointer_config["dropout"],
            cell_type=pointer_config["rnn_cell"], batch_first=False)

    def forward(self, question, question_mask,
                passage, passage_mask):
        question, passage = self.sentence_encoder(question, passage)
        passage = self.pair_encoder(passage, question, question_mask)
        passage = self.self_match_encoder(passage, passage, passage_mask)
        begin, end = self.pointer_net(question, question_mask, passage, passage_mask)
        return begin, end