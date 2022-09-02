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

import numpy as np
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import os
import copy
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
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
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
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg
def in_box_voxel(gtbox,voxels_cood1):
    voxels_cood = copy.deepcopy(voxels_cood1)
    # print(voxels_cood[:,0:3])
    voxels_cood[:,]=voxels_cood[:,] - gtbox[0:3] #z,y,x
    # print(voxels_cood[:,0:3])
    # print(gtbox)
    # print(torch.sin(gtbox[3]) )
    # print(torch.cos(gtbox[3]) )
    voxels_cood_y = voxels_cood[:,2]*torch.sin(gtbox[3]) \
    + voxels_cood[:,1]*torch.cos(gtbox[3])
    voxels_cood_x = -voxels_cood[:,1]*torch.sin(gtbox[3]) \
    + voxels_cood[:,2]*torch.cos(gtbox[3])
    # print(voxels_cood_x)
    voxels_cood[:,1] = voxels_cood_y
    voxels_cood[:,2] = voxels_cood_x
    # import ipdb; ipdb.set_trace()
    # in_bool1 = (voxels_cood[:,1] <= gtbox[5]/2) & (voxels_cood[:,1] >= -gtbox[5]/2) 
    # in_bool2 = (voxels_cood[:,2] <= gtbox[6]/2) & (voxels_cood[:,2] >= -gtbox[6]/2)
    # in_bool3 = (voxels_cood[:,0]<=gtbox[4]/2)&(voxels_cood[:,0]>= -gtbox[4]/2)
    in_bool =   (voxels_cood[:,1] <= gtbox[5]/2) & (voxels_cood[:,1] >= -gtbox[5]/2) \
    &(voxels_cood[:,2] <= gtbox[6]/2) & (voxels_cood[:,2] >= -gtbox[6]/2)\
    &(voxels_cood[:,0]<=gtbox[4]/2)&(voxels_cood[:,0]>= -gtbox[4]/2)
    # print(in_bool.sum().item())
    # print(in_bool1.sum().item())
    # print(in_bool2.sum().item())
    # print(in_bool3.sum().item())
    return in_bool

def main():
    args, cfg = parse_config()
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
    if not os.path.exists(os.path.join(output_dir, 'tools')):
        shutil.copytree('../tools', os.path.join(output_dir,'tools'))
    if not os.path.exists(os.path.join(output_dir, 'pcdet')):
        # exclude the ops(54M)
        shutil.copytree('../pcdet/datasets', os.path.join(output_dir,'./pcdet/datasets'))
        shutil.copytree('../pcdet/models', os.path.join(output_dir,'./pcdet/models'))
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
        os.system('cp %s %s' % (args.cfg_file, output_dir))

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
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)
    model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': train_set.point_feature_encoder.num_point_features,
            'num_point_features': train_set.point_feature_encoder.num_point_features,
            'grid_size': train_set.grid_size,
            'point_cloud_range': train_set.point_cloud_range,
            'voxel_size': train_set.voxel_size
        }
    for cur_it in range(total_it_each_epoch):
        batch_dict = next(dataloader_iter)
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
                continue
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        pc_range = model_info_dict['point_cloud_range']
        voxel_size = model_info_dict['voxel_size']
        voxel_coods = batch_dict['voxel_coords']
        # print(pc_range)
        # print(voxel_size)
        # pc_range = torch.tensor(pc_range)
        # voxel_size = torch.tensor(voxel_size)
        gt_boxes = batch_dict['gt_boxes']
        coor_centerx = (gt_boxes[:,:,0] - pc_range[0]) / voxel_size[0] 
        coor_centery = (gt_boxes[:,:,1] - pc_range[1]) / voxel_size[1] 
        coor_centerz = (gt_boxes[:,:,2] - pc_range[2]) / voxel_size[2] 
        coor_x = (gt_boxes[:,:,3]) / voxel_size[0] 
        coor_y = (gt_boxes[:,:,4]) / voxel_size[1] 
        coor_z = (gt_boxes[:,:,5]) / voxel_size[2] 
        # rotation = (-(gt_boxes[:,:,6]+np.pi/2))%np.pi
        rotation = gt_boxes[:,:,6]
        keep_index_list_total =[]
        f = open ("/home/nfs_data/lupu/CenterPoint-KITTI/gt_ratio",'a+')
        for i in range(batch_dict['batch_size']):
            label = gt_boxes[i,:,7]
            index_list = torch.nonzero(label.int()!= 0).squeeze()
            coor_centerx_batch = torch.index_select(coor_centerx[i],0,index_list)
            coor_centery_batch = torch.index_select(coor_centery[i],0,index_list)
            coor_centerz_batch = torch.index_select(coor_centerz[i],0,index_list)
            coor_x_batch = torch.index_select(coor_x[i],0,index_list)
            coor_y_batch = torch.index_select(coor_y[i],0,index_list)
            coor_z_batch = torch.index_select(coor_z[i],0,index_list)
            rotation_batch = - torch.index_select(rotation[i],0,index_list)
            coor_box =torch.stack((coor_centerz_batch,coor_centery_batch,coor_centerx_batch,rotation_batch,
                                                                    coor_z_batch,coor_y_batch,coor_x_batch),dim=1)
            index_list = torch.nonzero(voxel_coods[:,0].int()== i).squeeze()
            voxel_coods_batch= torch.index_select(voxel_coods, 0, index_list)
            inbox_index_bool= ~(voxel_coods_batch[:,0].int()== i)
            # print(inbox_index_bool.sum().item())
            # print("------")
            # list_12 =[]
            for i in range(coor_box.size()[0]):
                # print(coor_box[i])
                in_bool= in_box_voxel(coor_box[i],voxel_coods_batch[:,1:])
                # print(in_bool.sum().item())
                # list_12.append(in_bool.sum().item())
                inbox_index_bool = inbox_index_bool | in_bool
                # print(inbox_index_bool.size())
                # print(voxel_coods_batch[:,1].size())
                # print(inbox_index_bool.sum().item())
                # save the vis 
            # sum_in=0
            # for i  in list_12:
            #     sum_in += i
            # print(sum_in)
            # inbox_index_list = torch.nonzero(inbox_index_bool).squeeze()
            # voxel_coods_batch_gt= torch.index_select( voxel_coods_batch, 0, inbox_index_list)
            # origin_coords = batch_dict['voxel_coords']
            # print(inbox_index_bool.size())
            # print(voxel_coods_batch[:,1].size())
            # print(inbox_index_bool.sum().item())
            # print(voxel_coods_batch[:,1].numel())
            # exit()
            # d = {}
            # d['origin_coords'] = voxel_coods_batch[:,1:]
            # d['origin_coords_gt'] = voxel_coods_batch_gt[:,1:]
            # d['gt_boxes'] = coor_box
            # torch.save(d, './visualization/ratio_{}.pth'.format(1))
            # exit()
            # if int(inbox_index_bool.sum().item()/voxel_coods_batch[:,1].numel()) == 1:
            print(inbox_index_bool.sum().item()/voxel_coods_batch[:,1].numel(),file =f)
    

if __name__ == '__main__':
    main()