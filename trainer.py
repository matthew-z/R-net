import json
import os
import sys
import time

import torch
from tensorboard_logger import configure, log_value
from torch import optim
from torch.autograd import Variable

from models.r_net import RNet
from utils.squad_eval import evaluate


class Trainer(object):
    def __init__(self, dataloader_train, dataloader_dev, char_embedding_config, word_embedding_config,
                 sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config,
                 resume=False, resume_snapshot_path="trained_model",
                 dev_dataset_path="./data/squad/dev-v1.1.json"):

        # for validate
        expected_version = "1.1"
        with open(dev_dataset_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            if dataset_json['version'] != expected_version:
                print('Evaluation expects v-' + expected_version +
                      ', but got dataset with v-' + dataset_json['version'],
                      file=sys.stderr)
            self.dev_dataset = dataset_json['data']

        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev
        if not resume:
            self.model = RNet(char_embedding_config, word_embedding_config, sentence_encoding_config,
                              pair_encoding_config, self_matching_config, pointer_config)
        else:
            if os.path.exists(resume_snapshot_path):
                self.model = torch.load(open(resume_snapshot_path, "rb"))
            else:
                raise FileNotFoundError(
                    "resume snapshot not found at: %s" % resume_snapshot_path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.parameters_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.Adadelta(self.parameters_trainable, rho=0.95)

        configure("log/0826", flush_secs=5)

    def train(self, epoch_num):
        step = 0
        for epoch in range(epoch_num):
            global_loss = 0.0
            global_acc = 0.0
            last_step = -1
            last_time = time.time()

            for batch_idx, batch_train in enumerate(self.dataloader_train):
                loss, acc = self._forward(batch_train)
                global_loss += loss.data[0]
                global_acc += acc
                self._update_param(loss)

                if step % 10 == 0:
                    used_time = time.time() - last_time
                    step_num = step - last_step
                    print("step %d / %d of epoch %d)" % (batch_idx,
                                                         len(self.dataloader_train), epoch), flush=True)
                    print("loss: ", global_loss / step_num, flush=True)
                    print("acc: ", global_acc / step_num, flush=True)
                    print("speed: %f examples/sec \n\n" %
                          (self.dataloader_train.batch_size * step_num / used_time), flush=True)

                    log_value('train/acc', global_acc / step_num, step)
                    log_value('train/loss', global_loss / step_num, step)

                    global_loss = 0.0
                    global_acc = 0.0
                    last_step = step
                    last_time = time.time()

                step += 1

            if epoch % 2 != 0:
                continue

            exact_match, f1 = self.eval()
            print("exact_match: %f)" % exact_match, flush=True)
            print("f1: %f)" % f1, flush=True)

            log_value('dev/f1', f1, step)
            log_value('dev/EM', exact_match, step)

            torch.save(self.model, "trained_model")
            if f1 > self.model.current_score:
                self.model.current_score = f1
                torch.save(self.model, "best_trained_model")

    def eval(self):
        self.model.eval()
        pred_result = {}
        for _, batch in enumerate(self.dataloader_dev):

            question_ids, words, questions, passages, passage_tokenized = batch
            words.to_variable(volatile=True)
            questions.to_variable(volatile=True)
            passages.to_variable(volatile=True)
            begin_, end_ = self.model(
                words, questions, passages)  # batch x seq

            _, pred_begin = torch.max(begin_, 1)
            _, pred_end = torch.max(end_, 1)

            pred = torch.stack([pred_begin, pred_end], dim=1)

            for i, (begin, end) in enumerate(pred.cpu().data.numpy()):
                ans = passage_tokenized[i][begin:end + 1]
                qid = question_ids[i]
                pred_result[qid] = " ".join(ans)
        self.model.train()
        return evaluate(self.dev_dataset, pred_result)

    def _forward(self, batch):

        _, words, questions, passages, answers, _ = batch
        batch_num = questions.tensor.size(0)

        words.to_variable()
        questions.to_variable()
        passages.to_variable()
        answers = Variable(answers)
        if torch.cuda.is_available():
            answers = answers.cuda()
        begin_, end_ = self.model(words, questions, passages)  # batch x seq
        assert begin_.size(0) == batch_num

        begin, end = answers[:, 0], answers[:, 1]
        loss = self.loss_fn(begin_, begin) + self.loss_fn(end_, end)

        _, pred_begin = torch.max(begin_, 1)
        _, pred_end = torch.max(end_, 1)

        exact_correct_num = torch.sum(
            (pred_begin == begin) * (pred_end == end))
        acc = exact_correct_num.data[0] / batch_num

        return loss, acc

    def _update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
