import os
import pickle

import torch

from trainer import Trainer
from utils import prepare_data, get_args


def main():
    prepare_data()
    args = get_args()
    is_debug = args.debug
    print("DEBUG Mode: ", "On" if is_debug else "Off")
    train_cache = "./data/cache/SQuAD%s.pkl" % ("_debug" if is_debug else "")
    dev_cache = "./data/cache/SQuAD_dev%s.pkl" % ("_debug" if is_debug else "")

    train_json = args.train_json
    dev_json = args.dev_json

    train = read_dataset(train_json, train_cache, is_debug)
    dev = read_dataset(dev_json, dev_cache, is_debug, split="dev")

    dev_dataloader = dev.get_dataloader(args.batch_size_dev)
    train_dataloader = train.get_dataloader(args.batch_size, shuffle=True)

    char_embedding_config = {"embedding_weights": train.cv_vec,
                             "padding_idx": train.PAD,
                             "update": args.update_char_embedding,
                             "bidirectional": args.bidirectional,
                             "cell_type": "gru", "output_dim": 300}

    word_embedding_config = {"embedding_weights": train.wv_vec,
                             "padding_idx": train.PAD,
                             "update": args.update_word_embedding}

    sentence_encoding_config = {"hidden_size": args.hidden_size,
                                "num_layers": args.num_layers,
                                "bidirectional": True,
                                "dropout": args.dropout, }

    pair_encoding_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    self_matching_config = {"hidden_size": args.hidden_size,
                            "num_layers": args.num_layers,
                            "bidirectional": args.bidirectional,
                            "dropout": args.dropout,
                            "gated": True, "mode": "GRU",
                            "rnn_cell": torch.nn.GRUCell,
                            "attn_size": args.attention_size,
                            "residual": args.residual}

    pointer_config = {"hidden_size": args.hidden_size,
                      "num_layers": args.num_layers,
                      "dropout": args.dropout,
                      "residual": args.residual,
                      "rnn_cell": torch.nn.GRUCell}

    trainer = Trainer(train_dataloader, dev_dataloader,
                      char_embedding_config, word_embedding_config,
                      sentence_encoding_config, pair_encoding_config,
                      self_matching_config, pointer_config, resume=args.resume,
                      resume_snapshot_path=args.resume_snapshot_path, dev_dataset_path=args.dev_json)

    trainer.train(args.epoch_num)


def read_dataset(json_file, cache_file, is_debug=False, split="train"):
    if os.path.isfile(cache_file):
        dataset = pickle.load(open(cache_file, "rb"))
    else:
        print("building %s dataset" % split)
        from dataset import SQuAD
        dataset = SQuAD(json_file, debug_mode=is_debug, split=split)
        pickle.dump(dataset, open(cache_file, "wb"))
    return dataset


if __name__ == "__main__":
    main()
