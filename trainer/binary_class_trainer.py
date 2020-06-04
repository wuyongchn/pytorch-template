import torch
from collections import OrderedDict
from base import BaseTrainer


class BinaryClassTrainer(BaseTrainer):
    def __init__(self, model, cfg, data_loader, optimizer, scheduler, criterion):
        super(BinaryClassTrainer, self).__init__(
            cfg, model, data_loader, optimizer, scheduler)
        self._meter = {'total': [0, 0], 'pred': [0, 0]}
        self.criterion = criterion

    def train_iter(self, *args):
        data, target = args
        pred = self.model(data)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def test_iter(self, *args):
        data, target = args
        pred = self.model(data)
        loss = self.criterion(pred, target)
        self.update_metric(pred, target)
        return {'loss': loss}

    def update_metric(self, logit, target):
        _, predict = torch.max(logit, 1)
        result = predict.data == target.data
        for idx in range(logit.size(0)):
            self._meter['total'][target[idx].item()] += 1
            if bool(result[idx].item()):
                self._meter['pred'][target[idx].item()] += int(result[idx].item())

    def compute_accuracy(self):
        self.accuracy = OrderedDict()
        self.accuracy['class#0'] = 1 - (self._meter['pred'][0] / (self._meter['total'][0] + 1e-7))
        self.accuracy['class#1'] = 1 - (self._meter['pred'][1] / (self._meter['total'][1] + 1e-7))
        self._meter = {'total': [0, 0], 'pred': [0, 0]}