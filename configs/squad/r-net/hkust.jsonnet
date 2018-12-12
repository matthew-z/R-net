local embedding_size = 500;
local hidden_size = 75;
local attention_size = 75;
local num_layers = 3;
local dropout = 0.3;
local bidirectional = true;

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
        // min_padding_length: 5,
      },
    },
  },

  train_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json',
  validation_data_path: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json',
  model: {
    type: 'r_net',
    share_encoder: true,
    text_field_embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          pretrained_file: 'https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz',
          embedding_dim: 300,
          trainable: false,
        },
        token_characters: {
          type: 'character_encoding',
          embedding: {
            num_embeddings: 262,
            embedding_dim: 8,
          },
          encoder: {
            type: 'gru',
            input_size: 8,
            hidden_size: 100,
            bidirectional: true,
            dropout: dropout,
          },
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
    batch_size: 128,
    // padding_noise: 0.2,
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
      lr: 0.5,
      rho: 0.95,
    },
  },
}
