import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import logging

import time
import pickle

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

import sys
sys.path.append('../')
from eval_utils.eval_utils import statistics_info


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter,cur_epoch, logger, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        lr_scheduler.step(accumulated_iter)

        batch['cur_it'] = cur_it
        batch['itera_each_epoch'] = total_it_each_epoch
        batch['cur_epoch'] = cur_epoch
        batch['total_iteration'] = cur_epoch*total_it_each_epoch + cur_it
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict,ret_dict = model_func(model, batch)
        # break
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        # train_mode=1 donot care about predictor_loss
        # if 'predictor_loss' in ret_dict.keys():
            # import ipdb; ipdb.set_trace()
            # disp_dict.update({'predictor_loss': ret_dict['predictor_loss']})
            # # statics_of_drop_voxel = {'voxels_number': model.module_list[1].voxels_number, 'drop_voxels_number':model.module_list[1].drop_voxels_number ,
            # # 'voxels_in_boxes_number':model.module_list[1].voxels_in_boxes_number ,'drop_voxels_in_boxes_number':model.module_list[1].drop_voxels_in_boxes_number ,}
            # loss += ret_dict['predictor_loss']
            # if model.module_list[1].skip_drop_voxel:
                # loss = ret_dict['predictor_loss']
        loss.backward()
        # clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)   # DEBUG_ONLY
        optimizer.step()

        accumulated_iter += 1

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()
            # print(statics_of_drop_voxel)
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    # d={}
    # d['loc_lss_list'] = model.dense_head.loc_lss_list
    # torch.save(model.module_list[1].save_dict, f"./visualization/predictor_epoch{}.pth")
    # model.dense_head.loc_lss_list
    # exit()
    return accumulated_iter

# TODO: the logic of training the predictor
def train_predictor_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter,cur_epoch,logger,tb_log=None, leave_pbar=False):

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        lr_scheduler.step(accumulated_iter)

        batch['cur_it'] = cur_it
        batch['itera_each_epoch'] = total_it_each_epoch
        batch['cur_epoch'] = cur_epoch
        batch['total_iteration'] = cur_epoch*total_it_each_epoch + cur_it
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        loss, tb_dict, disp_dict,ret_dict = model_func(model, batch)
        # break
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        assert 'predictor_loss' in ret_dict.keys()
        disp_dict.update({'predictor_loss': ret_dict['predictor_loss'].item()})

        loss_origin = loss
        loss = ret_dict['predictor_loss']
        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        if accumulated_iter % 10 == 0: # DEBUG_ONLY
            logger.info('at iter {}, loss: {:.3f}, predictor_loss: {:.3f}'.format(accumulated_iter,loss_origin,loss.item()))
            predictor_d = model.module_list[1].save_dict

            drop_rate_l0 = sum(predictor_d['n_drop_voxel_L0']) / sum(predictor_d['n_voxel_L0'])
            inbox_rate_l0 = sum(predictor_d['n_drop_voxel_inbox_L0']) / sum(predictor_d['n_drop_voxel_L0'])
            drop_rate_l1 = sum(predictor_d['n_drop_voxel_L1']) / sum(predictor_d['n_voxel_L1'])
            inbox_rate_l1 = sum(predictor_d['n_drop_voxel_inbox_L1']) / sum(predictor_d['n_drop_voxel_L1'])

            logger.info('Level0: drop_rate:{:.3f}, drop_inbox_rate:{}'.format(drop_rate_l0, inbox_rate_l0))
            logger.info('Level1: drop_rate:{:.3f}, drop_inbox_rate:{}'.format(drop_rate_l1, inbox_rate_l1))

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()
            # print(statics_of_drop_voxel)
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    # d={}
    # d['loc_lss_list'] = model.dense_head.loc_lss_list
    # torch.save(model.module_list[1].save_dict, f"./visualization/predictor_epoch{}.pth")
    # model.dense_head.loc_lss_list
    # exit()
    return accumulated_iter

def intermediate_eval(cfg, model, dataloader, epoch_id, logger, train_mode, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):

        # DEBUG_ONLY:
        # if i > 100:
        #     continue

        # if i < 471:
        #     progress_bar.update()
        #     continue
        #useless, large to make sure right drop-voxel
        cur_it = 1e10
        total_it_each_epoch = 1e10
        cur_epoch = 1e10
        batch_dict['cur_it'] = cur_it
        batch_dict['itera_each_epoch'] = total_it_each_epoch
        batch_dict['cur_epoch'] = cur_epoch
        batch_dict['total_iteration'] = cur_epoch*total_it_each_epoch + cur_it
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,train_mode=0,test_loader=None,cfg=None,logger=logging.getLogger()):  # only the intermediate_eval requires cfg_infos

    # unpack the list of the optimizers
    # noted that when train_mode=1, the input opitimizer is actually the optimizer_predictor(in train.py)
    # so the variable optimizer-predictor is only used for train-mode == 2
    if isinstance(optimizer,list):
        assert train_mode==2
        optimizer_predictor = optimizer[1]
        optimizer = optimizer[0]
        lr_scheduler_predictor = lr_scheduler[1]
        lr_scheduler = lr_scheduler
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler_predictor = lr_warmup_scheduler[1]
            lr_warmup_scheduler = lr_warmup_scheduler[0]
        else:
            lr_warmup_scheduler_predictor = None

    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if train_mode==0:
                accumulated_iter = train_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    # total_it_each_epoch=5, # DEBUG_ONLY: set total_it_each_epoch smaller for quick debu
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    cur_epoch = cur_epoch,
                    logger=logger,
                )
            elif train_mode==1:
                accumulated_iter = train_predictor_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter,
                    cur_epoch = cur_epoch,
                    logger=logger,
                )
            elif train_mode==2:
                # accmulutaed  is an int as cnter
                accumulated_iter = train_one_epoch(
                    model, optimizer, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter, 
                    cur_epoch = cur_epoch,
                    logger=logger,
                )

                train_predictor_one_epoch(
                    model, optimizer_predictor, train_loader, model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                    total_it_each_epoch=5,
                    # total_it_each_epoch=total_it_each_epoch,   # DEBUG_ONLY: set total_it_each_epoch smaller for quick debu
                    dataloader_iter=dataloader_iter, 
                    cur_epoch = cur_epoch,
                    logger=logger,
                )

            # model.module_list[1].train_epoch +=1
            # save trained model
            trained_epoch = cur_epoch + 1
            # ckpt_save_interval = 1  # DEBUG_ONLY
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                intermediate_eval(cfg, model, test_loader, trained_epoch, logger, train_mode, result_dir=ckpt_save_dir)

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # TODO: model-save-dict: SAVE the intermediate results
            # save every epoch: pack as a dict, each element is a list
            # mean within each epoch. append for each epoch
            # get the model.save_dict here and save to output log

            # statics_of_drop_voxel = {'voxels_number': model.module_list[1].voxels_number, 'drop_voxels_number':model.module_list[1].drop_voxels_number ,
            # 'voxels_in_boxes_number':model.module_list[1].voxels_in_boxes_number ,'drop_voxels_in_boxes_number':model.module_list[1].drop_voxels_in_boxes_number ,}
            # torch.save(statics_of_drop_voxel, f"./visualization/multiconv/predictor_r{model.radius}_wd{optim_cfg.WEIGHT_DECAY}_lr{optim_cfg.LR}_epoch{cur_epoch}.pth")
            save_d= {**model.module_list[3].save_dict,**model.module_list[1].save_dict, **model.save_dict}
            logger.info('saved at epoch:{}'.format(trained_epoch))
            # reload / mean*
            torch.save(save_d, os.path.join(ckpt_save_dir,'../','saved.pth'))


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
