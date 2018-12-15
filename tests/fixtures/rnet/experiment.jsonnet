local embedding_size = 10;
local hidden_size = 5;
local attention_size = 5;
local num_layers = 2;
local dropout = 0.3;
local bidirectional = true;
local rnn_type = 'gru';

local classifier = {
  input_dim: hidden_size * 2,
  num_layers: 1,
  hidden_dims: [
    5,
  ],
  activations: [
    'linear',
  ],
  dropout: [
    0.0,
  ],
};

{
  dataset_reader: {
    type: 'squad',
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
      token_characters: {
        type: 'characters',
        character_tokenizer: {
          byte_encoding: 'utf-8',
          start_tokens: [259],
          end_tokens: [260],
        },
        min_padding_length: 5,
      },
    },
  },


  train_data_path: 'tests/fixtures/data/squad.json',
  validation_data_path: 'tests/fixtures/data/squad.json',
  model: {
    type: 'r_net',
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          embedding_dim: 2,
          trainable: true,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            num_embeddings: 262,
            embedding_dim: 8,
          },
          encoder: {
            type: 'cnn',
            embedding_dim: 8,
            num_filters: 8,
            ngram_filter_sizes: [5],
          },
          dropout: 0.2,
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
      type: 'dynamic_pair_encoder',
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
    type: 'bucket',
    sorting_keys: [['passage', 'num_tokens'], ['question', 'num_tokens']],
    batch_size: 64,
    padding_noise: 0.0,

  },

  trainer: {
    num_epochs: 1,
    grad_norm: 5.0,
    patience: 10,
    validation_metric: '+em',
    cuda_device: -1,
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 2,
    },
    optimizer: {
      type: 'adam',
      betas: [0.9, 0.9],
    },
  },
}
