import os
import time
from tqdm import tqdm
import torch
import torch.cuda
import torch.nn as nn
from utils import log, signal_handler, ActionEnum
from abc import abstractmethod, ABC


class BaseTrainer(ABC):
    def __init__(self, cfg, model, data_loader, optimizer, scheduler):
        if 'gpu_id' not in cfg or cfg.gpu_id is None:
            self.gpu = 0
            log.info('PyTorch CPU mode')
            device_model = model
        elif isinstance(cfg.gpu_id, int):
            self.gpu = 1
            log.info('PyTorch single GPU mode')
            log.info('Using GPU ' + str(cfg.gpu_id))
            log.check_lt(cfg.gpu_id, torch.cuda.device_count(),
                         'GPU %d is not available' % cfg.gpu_id)
            log.info('GPU {}: {}'.format(
                cfg.gpu_id, torch.cuda.get_device_name(cfg.gpu_id)))
            torch.cuda.set_device(cfg.gpu_id)
            device_model = model.cuda(cfg.gpu_id)
        elif isinstance(cfg.gpu_id, list):
            self.gpu_id = 2
            log.info('PyTorch multiple GPU mode')
            log.info('Using GPU ' + str(cfg.gpu_id))
            max_gpu_idx = max(cfg.gpu_id)
            log.check_lt(max_gpu_idx, torch.cuda.device_count(),
                         'GPU %d is not available' % cfg.gpu_id)
            for gpu_idx in cfg.gpu_id:
                log.info('GPU {}: {}'.format(
                    gpu_idx, torch.cuda.get_device_name(cfg.gpu_id)))
            device_model = nn.DataParallel(model, device_ids=cfg.gpu_id).cuda()
        else:
            raise ValueError
        self.model = device_model
        self.gpu_id = cfg.gpu_id

        if isinstance(data_loader, dict):
            self.train_data_loader = data_loader['train'] \
                if 'train' in data_loader else None
            self.test_data_loader = data_loader['test'] \
                if 'test' in data_loader else None
        else:
            self.train_data_loader = data_loader
            self.train_data_loader = data_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = cfg.epochs
        self.epoch = 0
        self.iter = 0
        self.starting_iter = 0
        self.test_interval = cfg.test_interval if 'test_interval' in cfg else None
        self.checkpoint = cfg.checkpoint if 'checkpoint' in cfg else 1
        self.prefix = cfg.checkpoint_prefix if 'checkpoint_prefix' in cfg else ''
        self.display = cfg.display if 'display' in cfg else 20
        self.accuracy = None
        self.test_init = cfg.test_init if 'test_init' in cfg else True
        self.average_loss = cfg.average_loss if 'average_loss' in cfg else 1
        self.meter = None  # save some results to calculate accuracy
        self.losses = []
        self.smoothed_loss = 0
        self.iteration_time = 0
        self.iteration_last = 0
        self.action_request_function = signal_handler.signal2action
        self.requested_early_exit = False
        self._log_fmt = 'Epoch: {}, Iteration: {} ({:.3f} iters/s, ' \
                        '{:.1f}s/{} iters), loss = {:.3e}, lr = {:.2e}'

        if 'resume' in cfg and cfg.resume is not None:
            self.resume_checkpoint(cfg.resume)

    def train(self):
        self.init_training_params()
        while self.epoch < self.epochs:
            self.adjust_lr()
            if self.saving_point():
                self.save_checkpoint()
            if self.testing_point():
                self.test()
            if self.requested_early_exit:
                return
            self.train_epoch()
            if self.requested_early_exit:
                return
            self.epoch += 1

        if self.saving_point():
            self.save_checkpoint()
        if self.testing_point():
            self.test()
        if self.requested_early_exit:
            return
        log.info('Optimization done')

    def train_epoch(self):
        self.model.train()
        for data in self.train_data_loader:
            if isinstance(data, tuple):
                data = list(data)
            if self.gpu:
                for idx, value in enumerate(data):
                    data[idx] = data[idx].cuda(self.gpu_id)
            losses = self.train_iter(*data)
            log.check(isinstance(losses, dict),
                      'The value return by self.train_iter() should be a dict')

            loss = 0
            for key, value in losses.items():
                loss += value
            self.update_smoothed_loss(loss)

            if self.display and self.iter != 0 and self.iter % self.display == 0:
                lapse = (time.time() - self.iteration_time + 1e-7)
                per_s = float(self.iter - self.iteration_last) / lapse
                log.info(self._log_fmt.format(self.epoch, self.iter, per_s, lapse,
                                              self.display, self.smoothed_loss, self.get_lr()))
                self.iteration_time = time.time()
                self.iteration_last = self.iter
                out_idx = 0
                for key, value in losses.items():
                    log.info('    Train net output #{}: {} = {:.6f}'.format(
                        out_idx, key, value))
                    out_idx += 1
            self.iter += 1

            request = self.get_requested_action()
            if request == ActionEnum.CHECKPOINT:
                self.save_checkpoint()
            elif request == ActionEnum.STOP:
                self.requested_early_exit = True
                log.info('Optimization stopped early')
                break

    @abstractmethod
    def train_iter(self, *arg):
        raise NotImplementedError

    def init_training_params(self):
        self.starting_iter = self.iter
        self.iteration_last = self.iter
        self.losses = []
        self.smoothed_loss = 0
        self.iteration_time = time.time()

    def test(self):
        log.info('Epoch {}, Iteration {}, Testing net'.format(self.epoch, self.iter))
        self.model.eval()
        with torch.no_grad():
            test_loss = dict()
            count = 0
            for data in tqdm(iter(self.test_data_loader), leave=False,
                             total=len(self.test_data_loader)):
                if isinstance(data, tuple):
                    data = list(data)
                if self.gpu:
                    for i, value in enumerate(data):
                        data[i] = data[i].cuda(self.gpu_id)
                losses = self.test_iter(*data)
                log.check(isinstance(losses, dict),
                          'The value return by self.test_iter() should be a dict')
                for key, value in losses.items():
                    if count == 0:
                        test_loss.setdefault(key, 0)
                    else:
                        test_loss[key] += value
                if self.get_requested_action() == ActionEnum.STOP:
                    self.requested_early_exit = True
                    break
                count += 1

            total_loss = 0
            for key, value in test_loss.items():
                total_loss += value
            log.info('    Test total loss: {:.6f}'.format(total_loss / count))
            out_idx = 0
            for key, value in test_loss.items():
                log.info('    Train net output #{}: {} = {:.6f}'.format(
                    out_idx, key, value / count))
                out_idx += 1

            self.compute_accuracy()
            if self.accuracy:
                for key, value in self.accuracy.items():
                    log.info('    Train net output #{}: {} = {:.6f}'.format(
                        out_idx, key, value))
                    out_idx += 1

        if self.requested_early_exit:
            log.info('Testing stopped early')

    @abstractmethod
    def test_iter(self, *arg):
        raise NotImplementedError

    def reset_start(self):
        self.epoch = 0
        self.iter = 0

    def testing_point(self):
        if self.test_interval <= 0:
            return False
        elif self.epoch == 0:
            return self.test_init
        else:
            return self.epoch % self.test_interval == 0

    def saving_point(self):
        return self.epoch != 0 and self.epoch % self.checkpoint == 0

    def update_smoothed_loss(self, loss):
        if len(self.losses) < self.average_loss:
            self.losses.append(loss)
            size = float(len(self.losses) + 1e-7)
            self.smoothed_loss = (self.smoothed_loss * (size - 1) + loss) / size
        else:
            idx = (self.iter - self.starting_iter) % self.average_loss
            self.smoothed_loss += float(loss - self.losses[idx]) / self.average_loss
            self.losses[idx] = loss

    @abstractmethod
    def compute_accuracy(self):
        raise NotImplementedError

    def get_lr(self):
        return self.scheduler.get_lr()[0]

    def adjust_lr(self):
        self.scheduler.step()

    def resume_checkpoint(self, resume):
        log.check(os.path.isfile(resume), 'No resume file: %s' % resume)
        log.info('Loading resume file: %s' % resume)
        checkpoint = torch.load(resume)
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iteration']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        log.info('Loaded checkpoint {} (epoch {}, iteration {})'.format(
            resume, self.epoch, self.iter))
        self.model.eval()

    def save_checkpoint(self):
        path, _ = os.path.split(self.prefix)
        if path:
            log.check(os.path.isdir(path), '%s, does not exist' % path)
        model_state = self.model.state_dict() if self.gpu <= 1 \
            else self.model.module.state_dict()
        state = {
            'epoch': self.epoch,
            'iteration': self.iter,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        name = '{}_epoch({})_iter({}).pth'.format(
            self.prefix, self.epoch, self.iter)
        torch.save(state, name)
        log.info("Saving checkpoint: %s" % name)

    def get_requested_action(self):
        if self.action_request_function is not None:
            return self.action_request_function()
        else:
            return ActionEnum.NONE
