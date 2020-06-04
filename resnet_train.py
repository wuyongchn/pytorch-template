import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from config import cfg, config_str
from dataset import ClsDataset, TestingTransformer, TrainingTransformer
from model.binary_class_model import resnet34_2class
from trainer.binary_class_trainer import BinaryClassTrainer
from base import set_reproducible, ImageLoader
from utils import log


def main():
    set_reproducible(cfg.seed)
    log.info(config_str())
    train_dataset = ClsDataset(cfg.root.train, cfg.source.train,
                               TrainingTransformer(cfg.crop_size),
                               ImageLoader(cfg.new_size))
    test_dataset = ClsDataset(cfg.root.test, cfg.source.test,
                              TestingTransformer(cfg.crop_size),
                              ImageLoader(cfg.new_size))

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size.train,
        shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size.test,
        shuffle=False, num_workers=4, drop_last=False)

    model = resnet34_2class(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr_param.base_lr, momentum=0.9,
                                weight_decay=cfg.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu_id)
    scheduler = StepLR(optimizer, step_size=cfg.lr_param.stepsize, gamma=cfg.lr_param.gamma)
    trainer = BinaryClassTrainer(model, cfg,
                                 {'train': train_loader, 'test': test_loader},
                                 optimizer, scheduler, criterion)
    trainer.train()
    # trainer.test()


if __name__ == "__main__":
    main()