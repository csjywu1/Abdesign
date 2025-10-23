#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

# 添加Igformer路径到sys.path
import sys
sys.path.insert(0, 'Igformer/models1')

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig

def parse():
    parser = argparse.ArgumentParser(description='training')

    # data
    parser.add_argument('--train_set', type=str, help='path to train set')
    parser.add_argument('--valid_set', type=str, help='path to valid set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=200, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=100, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, choices=['igformer'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')

    # task setting
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # loss weights
    parser.add_argument('--sequence_loss_weight', type=float, default=1.0, help='weight for sequence loss')
    parser.add_argument('--structure_loss_weight', type=float, default=1.0, help='weight for structure loss')
    parser.add_argument('--docking_loss_weight', type=float, default=1.0, help='weight for docking loss')
    parser.add_argument('--pdev_loss_weight', type=float, default=1.0, help='weight for pdev loss')
    
    # dyMEAN specific options
    parser.add_argument('--backbone_only', action='store_true', help='only use backbone atoms')
    parser.add_argument('--fix_channel_weights', action='store_true', help='fix channel weights')
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='do not predict edge distances')
    parser.add_argument('--no_memory', action='store_true', help='do not use memory mechanism')

    return parser.parse_args()


def main(args):
    # 如果通过命令行参数传入，则使用命令行参数；否则使用默认值
    if not hasattr(args, 'train_set') or args.train_set is None:
        # 默认参数（用于直接运行train2.py）
        args.train_set = "all_data/RAbD/train_fixed.json"
        args.valid_set = "all_data/RAbD/valid_fixed.json"
        args.save_dir = os.path.join(os.getcwd(), "my_checkpoints/models_igformer")
        args.cdr = "H3"
        args.paratope = "H3"
        args.max_epoch = 1
        args.save_topk = 10
        args.batch_size = 1
        args.shuffle = True
        args.model_type = "igformer"
        args.embed_dim = 64
        args.hidden_size = 128
        args.k_neighbors = 9
        args.n_layers = 3
        args.iter_round = 3
        args.bind_dist_cutoff = 6.6
        args.gpus = [6]  # 单卡测试
        
        # 添加缺失的参数
        args.lr = 1e-3
        args.final_lr = 1e-4
        args.warmup = 0
        args.grad_clip = 1.0
        args.patience = 1000
        args.num_workers = 4
        args.struct_only = False
        args.backbone_only = False
        args.fix_channel_weights = False
        args.no_pred_edge_dist = False
        args.no_memory = False

    ########### load your train / valid set ###########
    if (len(args.gpus) > 1 and int(os.environ.get('LOCAL_RANK', 0)) == 0) or len(args.gpus) == 1:
        print_log(args)
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')

    train_set = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope)
    valid_set = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope)

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))

    if args.model_type == 'igformer':
        # 导入igformer相关的模块
        from trainer import IgformerTrainer as Trainer
        from models import IgformerModel
        model = IgformerModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   cdr_type=args.cdr, paratope=args.paratope,
                   backbone_only=args.backbone_only, fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist, keep_memory=not args.no_memory)
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)

    if len(args.gpus) > 1:
        # Multi-GPU training setup
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = max(1, int(args.batch_size / len(args.gpus)))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank

    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    
    # Move model to GPU (DDP will be handled in trainer)
    if len(args.gpus) > 1:
        model = model.to(f'cuda:{args.local_rank}')
    else:
        model = model.to(f'cuda:{args.gpus[0]}')
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)

# 多卡运行示例：
# GPU=2,3,4,5,6,7 bash scripts/train/train_igformer.sh scripts/train/configs/single_cdr_design_igformer.json
