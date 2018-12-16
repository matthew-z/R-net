local embedding_size = 768;
local hidden_size = 75;
local attention_size = 75;
local num_layers = 3;
local dropout = 0.3;
local bidirectional = true;

{
  dataset_reader: {
    type: 'squad_truncated',
    truncate_train_only: false,
    max_passage_len: 300,
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: 'bert-base-cased',
        do_lowercase: false,
        use_starting_offsets: false,
      },
    },
  },

  train_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json',
  validation_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json',

  model: {
    type: 'r_net',
    share_encoder: true,
    text_field_embedder: {
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: ['bert', 'bert-offsets'],
      },
      token_embedders: {
        bert: {
          type: 'bert-pretrained',
          pretrained_model: 'bert-base-cased',
        },
      },
    },


    question_encoder: {
      type: 'concat_rnn',
      input_size: embedding_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      bidirectional: bidirectional,
      dropout: dropout,
    },

    passage_encoder: {
      type: 'concat_rnn',
      input_size: embedding_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      bidirectional: bidirectional,
      dropout: dropout,
    },

    pair_encoder: {
      type: 'static_pair_encoder',
      memory_size: hidden_size * 2 * num_layers,
      input_size: hidden_size * 2 * num_layers,
      hidden_size: hidden_size,
      attention_size: attention_size,
      bidirectional: bidirectional,
      dropout: dropout,
      batch_first: true,

    },

    self_encoder: {
      type: 'static_self_encoder',
      memory_size: hidden_size * 2,
      input_size: hidden_size * 2,
      hidden_size: hidden_size,
      attention_size: attention_size,
      bidirectional: bidirectional,
      dropout: dropout,
      batch_first: true,

    },

    output_layer: {
      type: 'pointer_network',
      question_size: hidden_size * 2 * num_layers,
      passage_size: hidden_size * 2,
      attention_size: attention_size,
      dropout: dropout,
      batch_first: true,
    },
  },

  iterator: {
    type: 'basic',
    // sorting_keys: [['passage', 'num_tokens'], ['question', 'num_tokens']],
    batch_size: 64,
    // padding_noise: 0.2,
    // biggest_batch_first: true
  },

  trainer: {
    num_epochs: 120,
    num_serialized_models_to_keep: 5,
    grad_norm: 5.0,
    patience: 10,
    validation_metric: '+f1',
    cuda_device: [0],
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 3,
    },
    optimizer: {
      type: 'adadelta',
      lr: 1,
      rho: 0.95,
    },
  },
}
