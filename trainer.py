import time

import torch
from torch import optim
from torch.autograd import Variable

from r_net.model import RNet

class Trainer(object):
    def __init__(self, dataloader_train, char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config,
                 self_matching_config, pointer_config):

        self.dataloader_train = dataloader_train

        self.model = RNet(char_embedding_config, word_embedding_config, sentence_encoding_config,
                          pair_encoding_config, self_matching_config, pointer_config)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.parameters_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.Adadelta(self.parameters_trainable, rho=0.95)


    def train(self, epoch_num):
        for epoch in range(epoch_num):
            global_loss = 0.0
            global_acc = 0.0
            last_step = -1
            last_time = time.time()
            for step, batch_train in enumerate(self.dataloader_train):
                loss, acc = self._forward(batch_train)
                global_loss += loss.data[0]
                global_acc += acc
                self._update_param(loss)

                if step % 5 == 0:
                    used_time = time.time() - last_time
                    step_num = step - last_step
                    print("step %d / %d of epoch %d)" % (step, len(self.dataloader_train), epoch), flush=True)
                    print("loss: ", global_loss / step_num, flush=True)
                    print("acc: ", global_acc / step_num, flush=True)
                    print("speed: %f examples/sec \n\n" %
                          (self.dataloader_train.batch_size * step_num / used_time), flush=True)

                    global_loss = 0.0
                    global_acc = 0.0
                    last_step = step
                    last_time = time.time()

            torch.save(self.model, open("trained_model", "wb"))

    def _forward(self, batch):

        question_ids, words, questions, passages, answers, answers_texts = batch
        batch_num = questions.tensor.size(0)

        words.to_variable()
        questions.to_variable()
        passages.to_variable()
        answers = Variable(answers)
        if torch.cuda.is_available():
            answers = answers.cuda()

        begin_, end_ = self.model(words, questions, passages)  # seq x batch x 1
        assert begin_.size(0) == batch_num

        begin, end = answers[:, 0], answers[:, 1]
        loss = self.loss_fn(begin_, begin) + self.loss_fn(end_, end)

        _, pred_begin = torch.max(begin_, 1)
        _, pred_end = torch.max(end_, 1)

        exact_correct_num = torch.sum((pred_begin == begin) * (pred_end == end))
        acc = exact_correct_num.data[0] / batch_num

        return loss, acc

    def _update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()