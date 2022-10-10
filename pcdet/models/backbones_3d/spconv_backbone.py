from functools import partial

import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

from pcdet.models.dense_heads.centerpoint_head_single import GaussianFocalLoss


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,times, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.i=0

        # define what needs to be stored for save_dict
        self.save_dict={}
        save_variable_names = ['predictor_loss']
        for level_ in range(2): # levels
            for name in ['n_voxel','n_drop_voxel','n_voxel_inbox','n_drop_voxel_inbox']:  # multi-level varaibles to save
                save_variable_names.append(name+'_L{}'.format(level_))
        print(save_variable_names)
        for name in save_variable_names:
            self.save_dict[name] = []
        self.save_dict['maps'] = {}


        # self.train_epoch=0
        self.skip_drop_epoch = kwargs['SKIP_DROP_EPOCH']
        self.drop_ratio1=[]
        self.drop_ratio2=[]
        self.voxel_size = kwargs['voxel_size']
        self.point_cloud_range = kwargs['point_cloud_range']
        self.OUT_SIZE_FACTOR =  kwargs['OUT_SIZE_FACTOR']

        self.voxels_number = 0
        self.drop_voxels_number = 0
        self.voxels_in_boxes_number=0
        self.drop_voxels_in_boxes_number= 0



        # ----- the predictor related configs ------
        # TODO: feature-based predictor design
        # Input: [N, C] voxel features -> Project to BEV
        # Predictor: [Conv, FC]
        # Loss: MSE, regress the heatmap (possibly regs)
        # ------------------------------------------
        # (?): predictor applied during training/after training (warmup?)
        # [warmup]: only train predictor weights, no actual drop
        #           - how to feed iter-size in?
        # 1. voxel-drop-criterion (plus spatial priors), also loss calc logic here
        self.use_predictor = self.model_cfg['use_predictor'] if 'use_predictor' in self.model_cfg.keys() else False
         # make it a global switch for whether use the predictor or not, change it in `train.py`, reading the config and turn it on at certain epoch

        # -----------  CFGS for predictor ----------
        # self.predictor_warmup = False
        # self.mask_predictor_output = self.model_cfg['mask_predictor_output'] if 'mask_predictor_output' in self.model_cfg.keys() else False # mask-predictor output, so only the places with voxels give supervision
        self.mask_predictor_output = False
        self.predictor_loss_type = 'gaussian_focal'   # choose the predictor-loss type, could be in cfg
        self.drop_type = 'median'   # choose the predictor-loss type, could be in cfg
        self.skip_drop_voxel = True        # no drop-voxel
        self.predictor_type = 'multi_conv'
        if self.use_predictor:
            if self.predictor_type == 'multi_conv':
                num_layers = 3
                C_ins = [4*times*5,64,16]
                C_outs = [64,16,8]
                ks = [5,5,3]
                pool_strides = [8,4,2]
                predictor_ordered_dict = OrderedDict()
                self.pools = nn.ModuleList([spconv.SparseMaxPool3d(kernel_size=pool_stride) for pool_stride in pool_strides])
                for _ in range(num_layers):
                    predictor_ordered_dict['conv{}'.format(_+1)] = nn.Conv2d(C_ins[_],C_outs[_],ks[_],padding='same',bias=False)
                    predictor_ordered_dict['bn{}'.format(_+1)] = nn.BatchNorm2d(C_outs[_],affine=False)
                    predictor_ordered_dict['nonlinear{}'.format(_+1)] = nn.Sigmoid()
                self.predictor = nn.Sequential(predictor_ordered_dict)
            else:
                raise NotImplementedError
            # Input [x_conv3(max-channel-output)*Z-axis-size]
            # kernel-size should roughly be like the gaussian kernel(TODO: need deiciding)
            self.gt_heatmap = None
            self.predictor_loss = 0.
            self.predictor_loss_lambda = 1.e-3

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 1*times, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(1*times),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(1*times, 1*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(1*times, 2*times, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(2*times, 2*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(2*times, 2*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )
        self.conv_list = nn.ModuleList([self.conv1,self.conv2])
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(2*times, 4*times, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(4*times, 4*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(4*times, 4*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(4*times, 4*times, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(4*times, 4*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(4*times, 4*times, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(4*times, 8*times, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(8*times),
            nn.ReLU(),
        )
        self.num_point_features = 8*times

    # def predictor_forward(self, x):
        # # conduct the predictor forward
        # # Filter to generate the ``masked`` Conv2d
        # # x: [bs, ch*Z, W, H]
        # sparse_mask_ = (x.sum(dim=1).unsqueeze(1) != 0).int()
        # out = self.predictor(x)  # maybe calc loss here
        # out = out*sparse_mask_
        # return out

    def in_box_voxel(self,voxels,batch_dict,div_number):
        # voxels: Coords [N,4]
        # batch-dict
        # div-number: downsample rate
        voxel_coords = torch.cat([voxels[:,0:1] ,torch.div(voxels[:,1:],div_number,rounding_mode='floor')], dim=-1)
        point_cloud_range = self.point_cloud_range
        voxel_size = self.voxel_size
        pc_range = torch.tensor(point_cloud_range)
        voxel_size = torch.tensor(voxel_size)
        gt_boxes = batch_dict['gt_boxes']
        # INFO: a little dirty
        # ---------------------------------------------------------
        # we project the voxel into [200, 150]
        # howver, the gt-box and gt-heatmap is in [400, 300](a deconv upsample in 2d-backbone)
        # in order to align, we make both of it be in [200, 150] space, thus /self.OUT_SIZE_FACTOR/2
        # ----------------------------------------------------------
        coor_centerx = (gt_boxes[:,:,0] - pc_range[0]) / voxel_size[0] /self.OUT_SIZE_FACTOR/2
        coor_centery = (gt_boxes[:,:,1] - pc_range[1]) / voxel_size[1] /self.OUT_SIZE_FACTOR/2
        coor_centerz = (gt_boxes[:,:,2] - pc_range[2]) / voxel_size[2] /self.OUT_SIZE_FACTOR/2
        coor_x = (gt_boxes[:,:,3]) / voxel_size[0] /self.OUT_SIZE_FACTOR/2
        coor_y = (gt_boxes[:,:,4]) / voxel_size[1] /self.OUT_SIZE_FACTOR/2
        coor_z = (gt_boxes[:,:,5]) / voxel_size[2] /self.OUT_SIZE_FACTOR/2
        # rotation = (-(gt_boxes[:,:,6]+np.pi/2))%np.pi
        rotation = gt_boxes[:,:,6]
        num_inboxes = 0
        # keep_index_list_total =[]
        # voxel_batch_list = []
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
            index_list = torch.nonzero(voxel_coords[:,0].int()== i).squeeze(dim = 1)
            if index_list.shape[0]== 0 :
                continue
            batch_start = index_list[0]
            voxel_coords_batch= torch.index_select(voxel_coords, 0, index_list)
            coor_box1 = torch.unsqueeze(coor_box, dim=1)
            voxel_coords_batch1 = torch.unsqueeze(voxel_coords_batch[:,1:],dim=0)
            voxel_coords_batch1 = voxel_coords_batch1 - coor_box1[:,:,0:3]
            voxels_coord_y = voxel_coords_batch1[:,:,2]*torch.sin(coor_box1[:,:,3]) + voxel_coords_batch1[:,:,1]*torch.cos(coor_box1[:,:,3])
            voxels_coord_x = -voxel_coords_batch1[:,:,1]*torch.sin(coor_box1[:,:,3]) + voxel_coords_batch1[:,:,2]*torch.cos(coor_box1[:,:,3])
            voxel_coords_batch1[:,:,1]=voxels_coord_y
            voxel_coords_batch1[:,:,2] = voxels_coord_x
            in_bool1 = (voxel_coords_batch1[:,:,1] <= (coor_box1[:,:,5]+1)/2) & (voxel_coords_batch1[:,:,1] >= -(coor_box1[:,:,5]+1)/2) \
            &(voxel_coords_batch1[:,:,2] <= (coor_box1[:,:,6]+1)/2) & (voxel_coords_batch1[:,:,2] >= -(coor_box1[:,:,6]+1)/2)\
            &(voxel_coords_batch1[:,:,0]<=(coor_box1[:,:,4]+1)/2)&(voxel_coords_batch1[:,:,0]>=-(coor_box1[:,:,4]+1)/2)
            in_bool1 = torch.any(in_bool1, 0)
            num_inboxes += in_bool1.sum(-1)
        # print(num_inboxes/voxel_coords.shape[0])
        return num_inboxes

    def drop_voxel(self, x,frame_id,level,batch_dict):#sparse_mask_, conf_map, voxel_data, voxel_data_pooled)

        x_conv = self.conv_list[level](x)
        if torch.isnan(x.features).sum()>0:
            print('X Nan Here!')
            import ipdb; ipdb.set_trace()
        x_pool_ = self.pools[level](x_conv)  # use pooling to pool all-levels of feature to the last dim(conv_4); TODO: the max-pool also pools the Z-axis feature harshly

        if torch.isnan(x_conv.features).sum()>0:
            print('X-conv1 Nan Here!')
            import ipdb; ipdb.set_trace()
        x_pool = x_pool_.dense()  # use pooling to pool all-levels of feature to the last dim(conv_4); TODO: the max-pool also pools the Z-axis feature harshly
        N,C,D,W,H=x_pool.shape                # concat the Z-axis to channel-dims,2-d HeatMap: [N,C,W,H]
        x_pool1_bak = x_pool
        x_pool = x_pool.reshape([N,-1,W,H])
        pool_size = 4//(level+1)
        x_pool = torch.repeat_interleave(x_pool,pool_size,dim=1)
        sparse_mask_ = (x_pool.sum(dim=1).unsqueeze(1) != 0).int()


        # predictor forward
        if self.predictor_type=='multi_conv':
            out_ = self.predictor(x_pool)
            out = torch.mean(out_,dim=1).unsqueeze(1)    #[bs,1,W,H]
        else:
            raise NotImplementedError

        if self.mask_predictor_output:
            conf_map = out*sparse_mask_  # DEBUG: whether to mask grad of places that donot have voxels
        else:
            conf_map = out

        # DEBUG_ONLY: save heatmaps for certain scenes
        frames_to_save = ['000000','000285', '001532', '003376', '005875','000845', '007429', '002738', '002235']
        for frame_name in frames_to_save:
            if np.where(frame_id == frame_name)[0].any():
                idx_batch = int(np.where(frame_id == frame_name)[0])
                #self.save_dict['input'] = x_pool[0]
                #self.save_dict['pre-bn'] = out1[0]
                #self.save_dict['post-bn'] = out1_bn[0]
                #self.save_dict['post-nonlinear'] = out[0]
                #self.save_dict['sparse-mask'] = sparse_mask_[0]
                gt_heatmap = torch.sum(self.gt_heatmap[0],dim=1)
                gt_heatmap = F.max_pool2d(gt_heatmap, kernel_size=2)
                gt_heatmap = gt_heatmap.unsqueeze(1)
                self.save_dict['maps'][frame_name] = torch.stack([conf_map[idx_batch],gt_heatmap[idx_batch],sparse_mask_[idx_batch]],dim=0)  # [3, W, H]

                # self.save_dict['conf_map'][frame_name] = conf_map[idx_batch]
                # self.save_dict['gt_heatmap'][frame_name] = gt_heatmap[idx_batch]
                # self.save_dict['sparse_map'][frame_name] = sparse_mask_[idx_batch]


        # if frame_id[0]=='000000':
            # gt_heatmap = torch.mean(self.gt_heatmap[0],dim=1)
            # gt_heatmap = F.avg_pool2d(gt_heatmap, kernel_size=2) 
            # #self.save_dict['input'] = x_pool[0]
            # #self.save_dict['pre-bn'] = out1[0]
            # #self.save_dict['post-bn'] = out1_bn[0]
            # #self.save_dict['post-nonlinear'] = out[0]
            # self.save_dict['conf_map'] = conf_map[0]
            # #self.save_dict['sparse-mask'] = sparse_mask_[0]
            # self.save_dict['gt_heatmap'] = gt_heatmap[0]
            # # torch.save(self.save_dict, f"./visualization/predictor/predictor_GT.pth")
            # # exit()
        # if frame_id[1]=='000000':
            # gt_heatmap = torch.mean(self.gt_heatmap[0],dim=1)
            # gt_heatmap = F.avg_pool2d(gt_heatmap, kernel_size=2)
            # #self.save_dict['input'] = x_pool[1]
            # #self.save_dict['pre-bn'] = out1[1]
            # #self.save_dict['post-bn'] = out1_bn[1]
            # #self.save_dict['post-nonlinear'] = out[1]
            # self.save_dict['conf_map'] = conf_map[1]
            # #self.save_dict['sparse-mask'] = sparse_mask_[1]
            # self.save_dict['gt_heatmap'] = gt_heatmap[1]
            # # torch.save(self.save_dict, f"./visualization/predictor/predictor_GT.pth")
            # # exit()
            # # DEBUG_ONLY:
            # # conf_map1_sparse_ratio = (conf_map1 > 0).sum() / conf_map1.nelement()
            # # out_sparse_ratio = (out > 0).sum() / out.nelement()
            # # sparse_mask_1_sparse_ratio = (sparse_mask_1 > 0).sum() / sparse_mask_1.nelement()
            # # x_pool1_sparse_ratio = (x_pool1>0).sum() / x_pool1.nelement()

        
        # if self.skip_drop_voxel:
        #     if self.train_epoch<16:
        #         gt_heatmap = torch.mean(self.gt_heatmap[0],dim=1)
        #         gt_heatmap = F.avg_pool2d(gt_heatmap, kernel_size=2) 
        #         predictor_loss_func = torch.nn.MSELoss()
        #         self.predictor_loss += self.predictor_loss_lambda*predictor_loss_func(conf_map, gt_heatmap)  
        #         return x_conv,x_conv ,None

        # if self.predictor_warmup:
            # warmup stage, return undropped original data and empty data
            # return x_conv, x_conv,None

        # -------- The Loss Calucation of the conf-map ----------
        # process the gt-heatmap
        gt_heatmap = torch.sum(self.gt_heatmap[0],dim=1)   # mean would cause too-small peak value, use sum to maintain 1
        gt_heatmap = F.max_pool2d(gt_heatmap, kernel_size=2) # avgpool2d will make the max being 0.5

        if self.predictor_loss_type == 'mse':
            predictor_loss_func = torch.nn.MSELoss()
            self.predictor_loss += self.predictor_loss_lambda*predictor_loss_func(conf_map, gt_heatmap)
        # elif self.predictor_loss_type == 'bce':
            # predictor_loss_func = torch.nn.BCELoss(weight=None)
        elif self.predictor_loss_type == 'gaussian_focal':
            num_pos = batch_dict['gt_boxes'].shape[1]  # num-boxes 
            predictor_loss_func = GaussianFocalLoss(reduction='mean')
            self.predictor_loss += self.predictor_loss_lambda*predictor_loss_func(conf_map, gt_heatmap, avg_factor=max(num_pos, 1))
        else:
            raise NotImplementedError


        # ----- criterions, how to hard prune from feature-map ------
        # TODO: threshold determination maybe?
        # get drop-indexs: [K,4] from bev-feature torch.where
        # DEBUG: conf-map use relu as activation! therefor the min is 0, or not? need to try
        # DEBUG: how to deal with bs, DONOT div on bs, make it all be 0?
        # DEBUG: when designing drop_where_dummy, make sure not to include 0 for conf-value, cause the sparse bev value is 0, will result in empty point-cloud 
        conf_map_reshape = conf_map.reshape(-1)
        sparse_mask_reshape = sparse_mask_.reshape(-1)
        conf_map_where_not_sparse = torch.nonzero(sparse_mask_reshape)
        not_sparse_conf_map = torch.index_select(conf_map_reshape,0,conf_map_where_not_sparse.squeeze(1))
        # conf_map_none_zero = torch.index_select(conf_map)


        if self.drop_type == 'median':
            drop_where_dummy = torch.where(((conf_map <= not_sparse_conf_map.median()) & (sparse_mask_!=0 )))  # tuple of 4 (BS,CH,W,H)
        elif self.drop_type == 'max_based':
            drop_where_dummy = torch.where(((conf_map <= not_sparse_conf_map.abs().max()*0.7) & (sparse_mask_!=0 )))  # tuple of 4 (BS,CH,W,H)
        elif self.drop_type == 'constant':
            drop_where_dummy = torch.where(((conf_map <= 0.65) & (sparse_mask_!=0 )))  # tuple of 4 (BS,CH,W,H)
        else:
            raise NotImplementedError

        # drop_where_dummy = torch.where(((conf_map >= conf_map.max()*0.5) & (conf_map > 0)))  # tuple of 4 (BS,CH,W,H)
        # torch.where(conf_map > conf_map.max()*0.01) 
        # drop_where_dummy = torch.where((conf_map < conf_map.median()*0.01) & (sparse_mask_!=0 )) 
        drop_dense_idx = torch.stack([drop_where_ for drop_where_ in drop_where_dummy],dim=-1)

        # TODO: the actual drop process, drop the voxel
        # 1. find the correspondance (BEV - sparse-voxel)
        #   - spconv sparse tensor, given the coord(indice), get the id/feature, gather-em
        #   - use from-dense function*  z-axis info all-lost, stupid as fuck
        # 2. drop the voxels(may need re-init sparse-tensor) TODO: check the grad
        voxel_coords = x_conv.indices[:,torch.LongTensor([0,2,3])]
        drop_dense_idx = drop_dense_idx[:,torch.LongTensor([0,2,3])]   # [K, 3]
        K,_ = drop_dense_idx.shape
        N,_ = voxel_coords.shape

        GET_DROP_INDEX_SCHEME = 'seq'
        # GET_DROP_INDEX_SCHEME = 'seq'
        # import ipdb; ipdb.set_trace()
        if K == 0:
            # PASS: donot drop point
            return x_conv, x_conv,None
        else:
            if GET_DROP_INDEX_SCHEME == 'para':
                # FIXME: K could be very large, which result in huge mem usage
                # Scheme 1:  parallel's K's matching
                voxel_coords_ = voxel_coords
                voxel_coords = torch.cat([voxel_coords[:,torch.LongTensor([0])] ,torch.div(voxel_coords[:,1:],x_conv.spatial_shape[0]//x_pool_.spatial_shape[0],rounding_mode='floor')], dim=-1).unsqueeze(0)       # [1,N,3], be careful //8 with the bs will cause all bs-id to be 0
                drop_dense_idx = drop_dense_idx.unsqueeze(1)   # [K,1,3]
                # [K,N,3], VERY HUGE Tensor
                diff = ((voxel_coords - drop_dense_idx)==0).sum(-1)  # [K,N], DEBUG: could be large when drop lots of points*
                ori_idx = torch.where(diff == 3)[0]
                drop_idx = torch.where(diff == 3)[1]
                # print(torch.unique(ori_idx).shape, drop_dense_idx.shape)   # check whether all dropped pixel have points within(should be)

            elif GET_DROP_INDEX_SCHEME == 'seq':
                # Scheme 2: sequential K's processing
                voxel_coords = torch.cat([voxel_coords[:,torch.LongTensor([0])] ,torch.div(voxel_coords[:,1:],x_conv.spatial_shape[0]//x_pool_.spatial_shape[0],rounding_mode='floor')], dim=-1)       # [1,N,3], be careful //8 with the bs will cause all bs-id to be 0
                drop_idx = torch.cat([torch.where((drop_dense_idx[k_]==voxel_coords).sum(-1) == 3)[0] for k_ in range(K)])
            else:
                raise NotImplementedError

            idx = torch.arange(N, device=x_conv.features.device)
            drop_mask = torch.ones(N, device=x_conv.features.device)
            drop_mask[drop_idx] = 0

            kept_idx = torch.masked_select(idx, drop_mask.bool())
            drop_idx = torch.masked_select(idx, ~drop_mask.bool())
            drop_voxels_coord = torch.index_select(x_conv.indices,0,drop_idx)
            num_in_boxes =  self.in_box_voxel(x_conv.indices,batch_dict,x_conv.spatial_shape[0]//x_pool_.spatial_shape[0])
            dropnum_in_boxes = self.in_box_voxel(drop_voxels_coord,batch_dict,x_conv.spatial_shape[0]//x_pool_.spatial_shape[0])
            # self.voxels_number = voxel_coords.shape[0]
            # self.drop_voxels_number = drop_idx.numel()
            # self.voxels_in_boxes_number=num_in_boxes.item() 
            # self.drop_voxels_in_boxes_number= dropnum_in_boxes.item()

            # save-drop-voxel data
            self.save_dict['n_voxel_L{}'.format(level)].append(voxel_coords.shape[0])
            self.save_dict['n_drop_voxel_L{}'.format(level)].append(drop_idx.numel())
            self.save_dict['n_voxel_inbox_L{}'.format(level)].append(num_in_boxes.item())
            self.save_dict['n_drop_voxel_inbox_L{}'.format(level)].append(dropnum_in_boxes.item())

            if dropnum_in_boxes.item() / drop_idx.numel() > 0.5:
                if 'failed_heatmap' not in self.save_dict.keys():
                    self.save_dict['failed_heatmap'] = []
                self.save_dict['failed_heatmap'].append(torch.cat([gt_heatmap.unsqueeze(1), conf_map, sparse_mask_],dim=1))  # [gt_heatmap, conf_map]: [bs, 2, 200, 176]

            if self.skip_drop_voxel:
                if batch_dict['cur_epoch']<self.skip_drop_epoch:
                    return x_conv, x_conv ,None

            indices = torch.index_select(x_conv.indices,0,kept_idx)
            features = torch.index_select(x_conv.features,0,kept_idx)
            new_voxel_data = spconv.SparseConvTensor(
                    features=features,
                    indices=indices.int(),
                    spatial_shape=x_conv.spatial_shape,
                    batch_size=x_conv.batch_size,
                    )

            # if level == 0:
                # self.drop_ratio1.append(new_voxel_data.features.shape[0]/x_conv.features.shape[0])
            # if level == 1:
                # self.drop_ratio2.append(new_voxel_data.features.shape[0]/x_conv.features.shape[0])
            return new_voxel_data, x_conv,drop_idx

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor) #may cause nan

        # debug-only
        # print('Cur input feature',x.features[0,:])  # [N,C]: reproducibility test, failed
        # print('\n',batch_dict['frame_id'],batch_dict['voxel_coords'][:3,:])
        # import ipdb; ipdb.set_trace()

        if self.use_predictor:
            self.predictor_loss = 0. # re-init the predictor loss after each iter
            # print('Sparse Ratio: conf_map:{:.3f}, predictor-out:{:.3f}, sparse-mask:{:.3f}, x_pool1 {:.3f},x-conv1-max:{}'\
                    # .format(conf_map1_sparse_ratio, out_sparse_ratio, sparse_mask_1_sparse_ratio, x_pool1_sparse_ratio, x_conv1.features.abs().max()))
            # x_pool1, sparse_map1 = self.drop_voxel(sparse_mask_1, conf_map1, x_conv1, x_pool1_)
            x_pool1,x_conv1 ,sparse_map1 = self.drop_voxel(x, batch_dict['frame_id'], 0,batch_dict)
            x_pool2,x_conv2 ,sparse_map2 = self.drop_voxel(x_pool1, batch_dict['frame_id'], 1,batch_dict)
            # print(x_conv1.features.shape[0], x_pool1.features.shape[0])
            self.save_dict['predictor_loss'].append(self.predictor_loss)

            x_conv3 = self.conv3(x_pool2)
            x_conv4 = self.conv4(x_conv3)


            # TODO: training logic of the predictor (without actual prune)  @lupu
            # one shared weights predictor for all-levels?(should be, same BEV-level input)
            # get-the gt-boxes and re-generate the gaussian-ball(with larger radius)
            # DEFINE: loss between gt and predictor-output, MSE-loss? or other pixel-wise loss like used in SR?


            # =====================================================================
            # WRONG: make batch-dim also // 4, stupid 
            # spatial_shape_ = x_conv1.spatial_shape // 4 # break-batch-info
            # indices_ = x_conv1.indices // 4
            # x_pool2 = spconv.SparseConvTensor(
                # features=x_conv1.features,
                # indices=indices_,
                # spatial_shape=spatial_shape_,
                # batch_size=batch_size
            # )
            # # print(x_test.indices.max())
            # print(x_pool1.features.shape, x_pool2.features.shape)

        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)




        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)#出现没有元素

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
