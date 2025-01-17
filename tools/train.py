import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import os
import warnings
import copy
# os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    # parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--times', type=int, default=16, help='3D backbone channel. For VoxelBackBone8x, default is 16')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu-id')
    parser.add_argument('--train_mode', type=int, default=0, help='0: model weight only; 1: predictor weight only; 2: joint training')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    cfg.OPTIMIZATION['train_mode']=args.train_mode  # feed the train_mode into the optim_cfg
    train_mode = args.train_mode
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # bakup the training files
    if os.path.exists(os.path.join(output_dir, 'tools')):
        shutil.rmtree(os.path.join(output_dir, 'tools'))
    shutil.copytree('../tools', os.path.join(output_dir,'tools'), ignore=shutil.ignore_patterns('scripts','visualization','*.pth'))

    if os.path.exists(os.path.join(output_dir, 'pcdet')):
        # exclude the ops(54M)
        shutil.rmtree(os.path.join(output_dir, './pcdet'))

    shutil.copytree('../pcdet/datasets', os.path.join(output_dir,'./pcdet/datasets'))
    shutil.copytree('../pcdet/models', os.path.join(output_dir,'./pcdet/models'))
    shutil.copy('../pcdet/models/backbones_3d/spconv_backbone.py', os.path.join(output_dir,'./'))
    shutil.copytree('../pcdet/utils', os.path.join(output_dir,'./pcdet/utils'))


    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        # save the config filename as "config.yaml" for better loading
        os.system('cp %s %s' % (args.cfg_file, os.path.join(output_dir, "config.yaml")))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, # DEBUG_ONLY
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )


    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, times = args.times)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # optimizer-predictor should have different OPTIMIZATION PARMAS during train_mode==2
    # during train_mode=1, cfg.OPTIMIZER to train predictor
    if train_mode == 0:
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        optimizer_predictor = None
    elif train_mode == 1:
        assert model.module_list[1].use_predictor
        print(model.module_list[1].predictor)
        optimizer_predictor = build_optimizer(model.module_list[1].predictor, cfg.OPTIMIZATION)   # name for a new optimizer to avoid load ckpt into the predictor-optimizer
        optimizer = None
    elif train_mode == 2:
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        optimizer_predictor = build_optimizer(model.module_list[1].predictor, cfg.OPTIMIZATION_PREDICTOR)
    else:
        raise NotImplementedError

    if model.debug:
        optimizer = build_optimizer(copy.deepcopy(model), cfg.OPTIMIZATION)
        # im not sure what these are for
        import ipdb; ipdb.set_trace()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    # not in-use for now
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    if train_mode == 0:
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
        lr_scheduler_predictor, lr_warmup_scheduler_predictor = None, None
    elif train_mode == 1:
        lr_scheduler, lr_warmup_scheduler = None, None
        lr_scheduler_predictor, lr_warmup_scheduler_predictor = build_scheduler(
            optimizer_predictor, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
    elif train_mode == 2:
        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )
        lr_scheduler_predictor, lr_warmup_scheduler_predictor = build_scheduler(
            optimizer_predictor, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION_PREDICTOR
        )

    # -----------------------start training---------------------------
    logger.info('**********************Start training with train_mode:%s %s/%s(%s)**********************'
                % (train_mode, cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if train_mode == 0:
        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            train_mode=train_mode,
            test_loader=test_loader, # feed in the testloader for valid
            cfg=cfg,
            logger=logger,
        )
    elif train_mode == 1:
        train_model(
            model,
            optimizer_predictor,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler_predictor,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler_predictor,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            train_mode=train_mode,
            test_loader=test_loader,
            cfg=cfg,
            logger=logger,
        )
    elif train_mode == 2:
        train_model(
            model,
            [optimizer,optimizer_predictor],
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=[lr_scheduler, lr_scheduler_predictor],
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=[lr_warmup_scheduler, lr_warmup_scheduler_predictor],
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            train_mode=train_mode,
            test_loader=test_loader,
            cfg=cfg,
            logger=logger,
        )
        import ipdb; ipdb.set_trace()

    else:
        raise NotImplementedError

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
