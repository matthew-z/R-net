import os

import torch
from torch.autograd import Variable
import time
from model import WordEmbedding, RNet

import pickle
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def main():
    start_time = time.time()
    DEBUG = True
    dataset_cache = "./data/cache/SQuAD%s.pkl" % ("_debug" if DEBUG else "")

    if os.path.isfile(dataset_cache):
        dataset = pickle.load(open(dataset_cache,"rb"))
    else:

        train_json = "./data/squad/train-v1.1.json"
        from dataset import SQuAD
        dataset = SQuAD(train_json, debug_mode=DEBUG)
        pickle.dump(dataset, open(dataset_cache, "wb"))

    dataloader = dataset.get_dataloader(5, shuffle=True)

    char_embedding_config = {"embedding_weights": dataset.cv_vec,
                             "padding_idx": dataset.PAD,
                             "update": True, "bidirectional": True,
                             "cell_type": "gru", "output_dim": 300}

    word_embedding_config = {"embedding_weights": dataset.wv_vec,
                             "padding_idx": dataset.PAD,
                             "update": False}

    sentence_encoding_config = {"hidden_size": 75, "num_layers": 3,
                                "bidirectional": True,
                                "dropout": 0.2, }

    pair_encoding_config = {"hidden_size": 75, "num_layers": 3, "bidirectional": True,
                            "dropout": 0.2, "gated": True, "mode":"GRU",
                            "rnn_cell": torch.nn.GRUCell, "attn_size":75, "residual": False}

    self_matching_config = {"hidden_size": 75, "num_layers": 3, "bidirectional": True,
                            "dropout": 0.2, "gated": True, "mode":"GRU",
                            "rnn_cell": torch.nn.GRUCell, "attn_size":75, "residual": False}

    pointer_config = {}

    model = RNet(char_embedding_config, word_embedding_config,sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config)

    for batch in dataloader:
        question_ids, words, questions, contexts, answers, answers_texts = batch

        words.to_variable()
        questions.to_variable()
        contexts.to_variable()

        predict = model(words, questions, contexts)
        break

    print("finished in %f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
    # import random
    # inputs = Variable(torch.randn(20, 5, 50))
    # lengths = []
    # for _ in range(5):
    #     lengths.append(random.randint(1, 16))
    #
    # lengths.sort(reverse=True)
    # pack = pack_padded_sequence(inputs, lengths)
    #
    # rnn = torch.nn.RNN(50, 100)
    #
    # rnn(pack)

