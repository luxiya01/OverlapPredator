from argparse import ArgumentParser
import os, torch, time, shutil, json,glob, argparse, shutil
import numpy as np
from easydict import EasyDict as edict
import wandb

from datasets.dataloader import get_dataloader, get_datasets
from models.architectures import KPFCNN
from lib.utils import setup_seed, load_config
from lib.tester import get_trainer
from mbes_tester import MBESTester
from lib.loss import MetricLoss
from configs.models import architectures

from mbes_data.datasets.mbes_data import get_multibeam_datasets

from torch import optim
from torch import nn
setup_seed(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mbes_config', type=str, default='mbes_data/configs/mbesdata_test_meters.yaml', help='Path to config file')
    parser.add_argument('--network_config', type=str, help= 'Path to the config file.',
                        default='network_configs/mbes_kitti.yaml')
    args = parser.parse_args()

    # load configs
    mbes_config = edict(load_config(args.mbes_config))
    network_config = edict(load_config(args.network_config))

    config = mbes_config
    for k, v in network_config.items():
        if k not in config:
            config[k] = v

    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']
    config['dataset_type'] = 'multibeam_npy_for_overlap_predator'

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    raw_config = edict(config)
    config.raw_config = raw_config
    
    # model initialization
    config.architecture = architectures[config.architecture]
    config.model = KPFCNN(config)   

    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    if config.dataset == 'multibeam':
        train_set, val_set, benchmark_set = get_multibeam_datasets(config)
    else:
        train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits
                                        )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)
    
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)

    name=config.exp_dir
    run = wandb.init(project='mbes-Predator', name=name,
                    config=config)
    wandb.tensorboard.patch(root_logdir=config.tboard_dir)
    trainer = MBESTester(config)
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()        
    wandb.finish()