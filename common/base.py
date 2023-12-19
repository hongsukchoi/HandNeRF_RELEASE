import os
import os.path as osp
import math
import glob
import abc
import numpy as np
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.parallel.data_parallel import DataParallel

from timer import Timer
from logger import colorlogger
from config import cfg
from model import get_model
from dataset import MultipleDatasets

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

        self.stats = {}

    @abc.abstractmethod
    def make_batch_generator(self):
        return

    @abc.abstractmethod
    def make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')
        self.train_objects = cfg.target_objects

    def get_optimizer(self, model):
        module_params_list = []
        module_name_list_string = ''
        for name, module in model.module.named_children():  # iterate immediate child modules
            if 'clip' in name:
                continue
            lr = cfg.lr_feature_encoder if name == 'feature_encoder' else cfg.lr
            params = {'params': module.parameters(), 'lr': lr}
            module_params_list.append(params)
            module_name_list_string += f'{name} / '

        optimizer = torch.optim.Adam(module_params_list, lr=cfg.lr)
        # optimizer = torch.optim.Adam(model.module.nerf.parameters(), lr=cfg.lr)  # empirically better .. don't know

        self.logger.info(f'[Training] The parameters of {module_name_list_string}are added to the optimizer.')
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        # optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('[Training] Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("[Dataloader] Creating dataset...")

        trainset_loader = []
        for i in range(len(cfg.trainset)):
            exec(f'from {cfg.trainset[i]}.{cfg.trainset[i]} import {cfg.trainset[i]}')
            trainset_loader.append(eval(cfg.trainset[i])('train', self.train_objects))
            self.logger.info(f"[Dataloader] {cfg.trainset[i]} target object: {trainset_loader[-1].target_objects}")
            self.logger.info(f"[Dataloader] {cfg.trainset[i]} target subject: {trainset_loader[-1].target_subjects}")

            self.logger.info(f"[Dataloader] Length of {cfg.trainset[i]}: {len(trainset_loader[i])}")

        trainset_loader = MultipleDatasets(trainset_loader, make_same_len=cfg.trainset_make_same_len)
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, worker_init_fn=worker_init_fn, pin_memory=True)

    def make_model(self):
        # prepare network
        self.logger.info("[Training] Creating graph and optimizer...")
        model = get_model('train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()
        if cfg.clip_loss:
            model.module.clip_model.eval()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):
    def __init__(self, test_epoch):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.test_epoch = int(test_epoch)
        self.test_objects = cfg.target_objects
        self.test_scene = ''
    
    def make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("[Dataloader] Creating dataset...")

        exec(f'from {cfg.testset}.{cfg.testset} import {cfg.testset}')
        testset_loader = eval(cfg.testset)('test', self.test_objects, self.test_scene)
        self.logger.info(f"[Dataloader] {cfg.testset} target object: {testset_loader.target_objects}")


        self.logger.info(f"[Dataloader] Length of {cfg.testset}: {len(testset_loader)}")
            
        self.itr_per_epoch = math.ceil(len(testset_loader) / cfg.num_gpus / cfg.test_batch_size)
        self.batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

    def make_model(self):
        # prepare network
        self.logger.info("[Testing] Creating graph...")
        model = get_model(cfg.test_mode)
        model = DataParallel(model).cuda()

        # load checkpoint
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        self.logger.info('[Testing] Load checkpoint from {}'.format(model_path))
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=True)
        # if overfitting
        # model.train()

        model.eval()

        self.model = model
