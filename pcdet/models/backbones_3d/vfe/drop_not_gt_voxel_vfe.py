import torch

from .vfe_template import VFETemplate
import random
import math
import numpy as np
import copy

class DropNotGTVoxelVFE(VFETemplate):
    # The pre-prcocess-based Drop Voxel should be defined here
    # TODO: input interface for point-level feature.
    # INFO: however, there would be many ops within the 3d backbone inference process, how to feed in?
    # write another OP maybe?
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.dict = kwargs
    def get_output_feature_dim(self):
        return self.num_point_features

    def in_box_voxel(self,gtbox,voxels_cood):
        voxels_cood = copy.deepcopy(voxels_cood)
        voxels_cood[:,]=voxels_cood[:,] - gtbox[0:3] #z,y,x
        voxels_cood_y = voxels_cood[:,2]*torch.sin(gtbox[3]) 
        + voxels_cood[:,1]*torch.cos(gtbox[3])
        voxels_cood_x = -voxels_cood[:,1]*torch.sin(gtbox[3]) 
        + voxels_cood[:,2]*torch.cos(gtbox[3])
        voxels_cood[:,1] = voxels_cood_y
        voxels_cood[:,2] = voxels_cood_x
        in_bool =   (voxels_cood[:,1] <= gtbox[5]/2+1) & (voxels_cood[:,1] >= -gtbox[5]/2-1) \
        &(voxels_cood[:,2] <= gtbox[6]/2+1) & (voxels_cood[:,2] >= -gtbox[6]/2-1)\
        &(voxels_cood[:,0]<=gtbox[4]/2+1)&(voxels_cood[:,0]>=-gtbox[4]/2-1)
        return in_bool

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # import ipdb; ipdb.set_trace()
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        # ==== EXP: Oracle-1 of the DropVoxel: Randomly drop voxel outside the bbox====
        # TODO:
        #   - config of different dropping techniques
        #   - find the voxels outside the bbox(create an idx list)
        #   - random choice of the voxel-not-in-box
        #
        point_cloud_range = self.dict['point_cloud_range']
        voxel_size = self.dict['voxel_size']
        voxel_coods = batch_dict['voxel_coords']
        pc_range = torch.tensor(point_cloud_range)
        voxel_size = torch.tensor(voxel_size)
        gt_boxes = batch_dict['gt_boxes']
        coor_centerx = (gt_boxes[:,:,0] - pc_range[0]) / voxel_size[0] #再确认一下
        coor_centery = (gt_boxes[:,:,1] - pc_range[1]) / voxel_size[1] 
        coor_centerz = (gt_boxes[:,:,2] - pc_range[2]) / voxel_size[2] 
        coor_x = (gt_boxes[:,:,3]) / voxel_size[0] 
        coor_y = (gt_boxes[:,:,4]) / voxel_size[1] 
        coor_z = (gt_boxes[:,:,5]) / voxel_size[2] 
        ratation = (-(gt_boxes[:,:,6]+np.pi/2))%np.pi
        # ratation = gt_boxes[:,:,6]
        keep_index_List_total =[]
        for i in range(batch_dict['batch_size']):
            label = gt_boxes[i,:,7]
            index_list = torch.nonzero(label.int()!= 0).squeeze()
            coor_centerx_batch = torch.index_select(coor_centerx[i],0,index_list)
            coor_centery_batch = torch.index_select(coor_centery[i],0,index_list)
            coor_centerz_batch = torch.index_select(coor_centerz[i],0,index_list)
            coor_x_batch = torch.index_select(coor_x[i],0,index_list)
            coor_y_batch = torch.index_select(coor_y[i],0,index_list)
            coor_z_batch = torch.index_select(coor_z[i],0,index_list)
            ratation_batch = - torch.index_select(ratation[i],0,index_list)
            coor_box =torch.stack((coor_centerz_batch,coor_centery_batch,coor_centerx_batch,ratation_batch,
            coor_z_batch,coor_y_batch,coor_x_batch),dim=1)
            index_list = torch.nonzero(voxel_coods[:,0].int()== i).squeeze()
            batch_start = index_list[0]
            voxel_coods_batch= torch.index_select(voxel_coods, 0, index_list)
            inbox_index_bool= ~torch.nonzero(voxel_coods_batch[:,0].int()== i).squeeze()
            # outbox_index_list=[]
            # print(voxel_coods_batch)
            for i in range(coor_box.size()[0]):
                # print(coor_box[i])
                in_bool= self.in_box_voxel(coor_box[i],voxel_coods_batch[:,1:])
                inbox_index_bool = inbox_index_bool |in_bool
                # inbox_index_list.append(in_index)
            # inbox_index_list=torch.cat(inbox_index_list,dim = 0)
            # inbox_index_list = torch.unique(inbox_index_list)
            # exit()
            # outbox_index_list=torch.cat(outbox_index_list,dim = 0)
            # outbox_index_list = torch.unique(outbox_index_list)
            inbox_index_list = torch.nonzero(inbox_index_bool).squeeze()
            if self.model_cfg["VOXEL_PERCENT"]*index_list.size()[0]>inbox_index_list.size()[0]:
                outbox_index_list = torch.nonzero(~inbox_index_bool).squeeze()
                point_num = self.model_cfg["VOXEL_PERCENT"]*index_list.size()[0] - inbox_index_list.size()[0]
                out_keep_index = random.sample(outbox_index_list, int(point_num))
                out_keep_index = torch.tensor(out_keep_index, dtype=torch.int).cuda()
                keep_index_List=torch.cat([inbox_index_list,out_keep_index],dim=0)
            else:
                keep_index_List=random.sample(inbox_index_list.tolist(),
                    int(self.model_cfg["VOXEL_PERCENT"]*index_list.size()[0]))
                keep_index_List = torch.tensor(keep_index_List, dtype=torch.int).cuda()
            keep_index_List = keep_index_List+batch_start
            keep_index_List_total.append(keep_index_List)
        keep_index_List_total=torch.cat(keep_index_List_total,dim=0)
        voxel_coods= torch.index_select(voxel_coods, 0, keep_index_List_total)
        points_mean= torch.index_select(points_mean, 0, keep_index_List_total)
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coods.contiguous()
        return batch_dict

