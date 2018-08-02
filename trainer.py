import datetime
import json
import os
import shutil
import sys
import time

import torch
from tensorboard_logger import configure, log_value
from torch import optim
from torch.autograd import Variable
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
import models.r_net as RNet
from utils.squad_eval import evaluate
from utils.utils import make_dirs
from torch import nn
import numpy as np
from torch.nn import functional as F
def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    checkpoint_regular = os.path.join(path, filename)
    checkpint_best = os.path.join(path, best_filename)
    torch.save(state, checkpoint_regular)
    if is_best:
        shutil.copyfile(checkpoint_regular, checkpint_best)


class Trainer(object):
    def __init__(self, args, dataloader_train, dataloader_dev, char_embedding_config, word_embedding_config,
                 sentence_encoding_config, pair_encoding_config, self_matching_config, pointer_config, trainer_config):

        # for validate
        expected_version = "1.1"
        with open(args.dev_json) as dataset_file:
            dataset_json = json.load(dataset_file)
            if dataset_json['version'] != expected_version:
                print('Evaluation expects v-' + expected_version +
                      ', but got dataset with v-' + dataset_json['version'],
                      file=sys.stderr)
            self.dev_dataset = dataset_json['data']

        self.dataloader_train = dataloader_train
        self.dataloader_dev = dataloader_dev

        self.model = RNet.RNet(args, char_embedding_config, word_embedding_config, sentence_encoding_config,
                               pair_encoding_config, self_matching_config, pointer_config)
        self.parameters_trainable = list(
            filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = trainer_config["optimizer"](self.parameters_trainable, rho=0.95, lr=trainer_config["lr"])
        # self.optimizer = torch.optim.Adam(self.parameters_trainable)
        self.scheduler = trainer_config["scheduler"](self.optimizer, "max",factor=trainer_config["factor"])
        self.grad_norm = trainer_config["grad_norm"]
        self.best_f1 = 0
        self.step = 0
        self.start_epoch = args.start_epoch
        self.name = args.name
        self.start_time = datetime.datetime.now().strftime('%b-%d_%H-%M')
        self.debug=args.debug

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_f1 = checkpoint['best_f1']
                self.name = checkpoint['name']
                self.step = checkpoint['step']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_time = checkpoint['start_time']
                self.scheduler = checkpoint['scheduler']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
        else:
            self.name += "_" + self.start_time

        # use which device
        if torch.cuda.is_available():
            self.model = self.model.cuda(args.device_id)
        else:
            self.model = self.model.cpu()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        configure("log/%s" % (self.name), flush_secs=5)
        self.checkpoint_path = os.path.join(args.checkpoint_path, self.name)
        make_dirs(self.checkpoint_path)

    def train(self, epoch_num):
        for epoch in range(self.start_epoch, epoch_num):
            global_loss = 0.0
            global_acc = 0.0
            last_step = self.step - 1
            last_time = time.time()

            for batch_idx, batch_train in enumerate(self.dataloader_train):
                loss, acc = self._forward(batch_train)
                global_loss +=  loss.item()
                global_acc += acc.item()
                self._update_param(loss)

                if self.step % 100 == 0:
                    used_time = time.time() - last_time
                    step_num = self.step - last_step
                    print("step %d / %d of epoch %d)" % (batch_idx, len(self.dataloader_train), epoch), flush=True)
                    print("loss: %.3f"%(global_loss / step_num), flush=True)
                    print("acc: %.3f" % (global_acc / step_num * 100), flush=True)
                    speed = self.dataloader_train.batch_size * step_num / used_time
                    print("speed: %f examples/sec \n\n" %
                          (speed), flush=True)

                    log_value('train/EM', global_acc / step_num, self.step)
                    log_value('train/loss', global_loss / step_num, self.step)
                    log_value('train/speed', speed, self.step)

                    global_loss = 0.0
                    global_acc = 0.0
                    last_step = self.step
                    last_time = time.time()
                self.step += 1

            if not self.debug:
                with torch.no_grad():
                    exact_match, f1 = self.eval()

                print("exact_match: %f)" % exact_match, flush=True)
                print("f1: %f)" % f1, flush=True)
                log_value('dev/f1', f1, self.step)
                log_value('dev/EM', exact_match, self.step)

                if f1 > self.best_f1:
                    is_best = True
                    self.best_f1 = f1
                else:
                    is_best = False

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'step': self.step + 1,
                    'best_f1': self.best_f1,
                    'name': self.name,
                    'optimizer': self.optimizer.state_dict(),
                    'start_time': self.start_time,
                    'scheduler':self.scheduler
                }, is_best, self.checkpoint_path)



    def eval(self):
        self.model.eval()
        pred_result = {}




        for _, batch in enumerate(self.dataloader_dev):

            question_ids, question, question_char, question_mask, \
            passage, passage_char, passage_mask, passage_tokenized = batch
            begin_, end_ = self.model(question, question_char, question_mask, passage, passage_char, passage_mask)  # batch x seq
            probability = torch.bmm(F.softmax(begin_, dim=-1).unsqueeze(2), F.softmax(end_, dim=-1).unsqueeze(1)).cpu().numpy()
            mask = np.float32(np.fromfunction(lambda k, i, j: (i - j <= 15) & (i <= j), (1, probability.shape[1], probability.shape[2])))

            probability = mask * probability
            pred_begin = np.argmax(np.amax(probability, axis=2),axis=1)
            pred_end = np.argmax(np.amax(probability, axis=1),axis=1)

            for i, (begin, end) in enumerate(zip(pred_begin, pred_end)):
                ans = passage_tokenized[i][begin:end + 1]
                qid = question_ids[i]
                pred_result[qid] = " ".join(ans)
        self.model.train()
        em, f1 =  evaluate(self.dev_dataset, pred_result)
        self.scheduler.step(em)
        return em, f1

    def _forward(self, batch):
        qid, question, question_char, question_mask, \
        passage, passage_char, passage_mask, answer = batch

        batch_num = question.size(0)

        begin_, end_ = self.model(question, question_char, question_mask, passage, passage_char, passage_mask)  # batch x seq

        assert begin_.size(0) == batch_num
        if torch.cuda.is_available():
            answer = answer.cuda()
        begin, end = answer[:, 0], answer[:, 1]

        loss = self.loss_fn(begin_, begin) + self.loss_fn(end_, end)

        _, pred_begin = torch.max(begin_, 1)
        _, pred_end = torch.max(end_, 1)
        exact_correct_num = torch.sum(
            (pred_begin == begin) * (pred_end == end))
        em = exact_correct_num.float() / batch_num

        return loss, em

    def _update_param(self, loss):
        self.model.zero_grad()
        loss.backward()
        self._rescale_gradients()
        self.optimizer.step()


    def _rescale_gradients(self) -> None:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self.grad_norm:
            clip_grad_norm(self.model.parameters(), self.grad_norm)