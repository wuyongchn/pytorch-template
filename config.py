from easydict import EasyDict as edict

cfg = edict()
cfg.gpu_id = 0
cfg.seed = 1
cfg.test_interval = 1
cfg.display = 100
cfg.epochs = 10
cfg.checkpoint = 1
cfg.test_init = False
cfg.checkpoint_prefix = 'model.pth/resnet34'
cfg.resume = 'model.pth/'

cfg.new_size = 224  # 256
cfg.crop_size = 224
cfg.is_color = True

cfg.root = edict()
cfg.source = edict()


cfg.root.train = '/home/wuyong/Datasets/oulu-npu'
cfg.root.test = '/home/wuyong/Datasets/oulu-npu'
cfg.source.train = '/home/wuyong/Datasets/oulu-npu/protocol_1/train_binary_balance_0920.txt'
cfg.source.test = '/home/wuyong/Datasets/oulu-npu/protocol_1/test_binary_list.txt.d20'
cfg.batch_size = edict()
cfg.batch_size.train = 48
cfg.batch_size.test = 32

cfg.lr_param = edict()
cfg.lr_param.base_lr = 5e-4
cfg.lr_param.stepsize = 5
cfg.lr_param.gamma = 0.1
cfg.weight_decay = 1e-4


def config_str():
    buf = 'Training Config:\n'
    for key, value in cfg.items():
        buf += '\t' + str(key)
        if isinstance(value, dict):
            buf += ':\n'
            for k, v in value.items():
                buf += '\t\t{}: {}\n'.format(k, v)
        else:
            buf += ': ' + str(value) + '\n'
    return buf.rstrip('\n')
