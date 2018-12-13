from typing import Dict, Optional, List, Any

import torch
from allennlp.data import Vocabulary
from allennlp.models import BidirectionalAttentionFlow
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, SquadEmAndF1
from torch.nn.functional import nll_loss

from modules.pair_encoder.pair_encoder import AttentionEncoder
from modules.pointer_network.pointer_network import QAOutputLayer

@Model.register("r_net")
class RNet(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 question_encoder: Seq2SeqEncoder,
                 passage_encoder: Seq2SeqEncoder,
                 pair_encoder: AttentionEncoder,
                 self_encoder: AttentionEncoder,
                 output_layer: QAOutputLayer,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 share_encoder: bool = False):

        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.pair_encoder = pair_encoder
        self.self_encoder = self_encoder
        self.output_layer = output_layer

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()
        self.share_encoder = share_encoder
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        question_embeded = self.text_field_embedder(question)
        passage_embeded = self.text_field_embedder(passage)

        question_mask = get_text_field_mask(question).byte()
        passage_mask = get_text_field_mask(passage).byte()

        quetion_encoded = self.question_encoder(
            question_embeded, question_mask)
        
        if self.share_encoder:
            passage_encoded = self.question_encoder(passage_embeded, passage_mask)
        else:
            passage_encoded = self.passage_encoder(passage_embeded, passage_mask)

        passage_encoded = self.pair_encoder(
            passage_encoded, passage_mask, quetion_encoded, question_mask)
        passage_encoded = self.self_encoder(
            passage_encoded, passage_mask, passage_encoded, passage_mask)

        span_start_logits, span_end_logits = self.output_layer(
            quetion_encoded, question_mask, passage_encoded, passage_mask)

        # Calculating loss and making prediction
        # Following code is copied from allennlp.models.BidirectionalAttentionFlow
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)

        span_start_logits = util.replace_masked_values(
            span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(
            span_end_logits, passage_mask, -1e7)

        best_span = self.get_best_span(span_start_logits, span_end_logits)

        output_dict = {
            "span_start_logits": span_start_logits,
            "span_start_probs": span_start_probs,
            "span_end_logits": span_end_logits,
            "span_end_probs": span_end_probs,
            "best_span": best_span,
        }


        if span_start is not None:
            loss = nll_loss(util.masked_log_softmax(
                span_start_logits, passage_mask), span_start.squeeze(-1))
            self._span_start_accuracy(
                span_start_logits, span_start.squeeze(-1))
            loss += nll_loss(util.masked_log_softmax(span_end_logits,
                                                     passage_mask), span_end.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end.squeeze(-1))
            self._span_accuracy(best_span, torch.stack(
                [span_start, span_end], -1))
            output_dict["loss"] = loss

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            batch_size = question_embeded.size(0)
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_span = tuple(best_span[i].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output_dict['best_span_str'].append(best_span_string)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {'start_acc': self._span_start_accuracy.get_metric(reset),
                'end_acc': self._span_end_accuracy.get_metric(reset),
                'span_acc': self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }

    @staticmethod
    def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size
        span_start_argmax = [0] * batch_size
        best_word_span = span_start_logits.new_zeros((batch_size, 2), dtype=torch.long)

        span_start_logits = span_start_logits.detach().cpu().numpy()
        span_end_logits = span_end_logits.detach().cpu().numpy()

        for b in range(batch_size):  # pylint: disable=invalid-name
            for j in range(passage_length):
                val1 = span_start_logits[b, span_start_argmax[b]]
                if val1 < span_start_logits[b, j]:
                    span_start_argmax[b] = j
                    val1 = span_start_logits[b, j]

                val2 = span_end_logits[b, j]

                if val1 + val2 > max_span_log_prob[b]:
                    best_word_span[b, 0] = span_start_argmax[b]
                    best_word_span[b, 1] = j
                    max_span_log_prob[b] = val1 + val2
        return best_word_span
