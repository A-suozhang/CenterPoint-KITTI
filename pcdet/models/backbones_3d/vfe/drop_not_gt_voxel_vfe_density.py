import torch

from .vfe_template import VFETemplate
import random
import math
import numpy as np
import copy

class DropNotGTVoxelDensityVFE(VFETemplate):
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

    def in_box_voxel(self,gtbox,voxels_coord):
        voxels_coord = copy.deepcopy(voxels_coord)
        voxels_coord[:,]=voxels_coord[:,] - gtbox[0:3] #z,y,x
        voxels_coord_y = voxels_coord[:,2]*torch.sin(gtbox[3]) +\
         voxels_coord[:,1]*torch.cos(gtbox[3])
        voxels_coord_x = -voxels_coord[:,1]*torch.sin(gtbox[3]) +\
         voxels_coord[:,2]*torch.cos(gtbox[3])
        voxels_coord[:,1] = voxels_coord_y
        voxels_coord[:,2] = voxels_coord_x
        # import ipdb; ipdb.set_trace()
        in_bool =   (voxels_coord[:,1] <= gtbox[5]/2+1) & (voxels_coord[:,1] >= -gtbox[5]/2-1) \
        &(voxels_coord[:,2] <= gtbox[6]/2+1) & (voxels_coord[:,2] >= -gtbox[6]/2-1)\
        &(voxels_coord[:,0]<=gtbox[4]/2+1)&(voxels_coord[:,0]>=-gtbox[4]/2-1)
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
        origin_coords = batch_dict['voxel_coords']
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
        voxel_coords = batch_dict['voxel_coords']
        pc_range = torch.tensor(point_cloud_range)
        voxel_size = torch.tensor(voxel_size)
        voxel_range = ((pc_range[3:]-pc_range[0:3])/voxel_size).cuda()
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
                                                                    coor_z_batch,coor_y_batch,coor_x_batch),dim=1).cuda()
            index_list = torch.nonzero(voxel_coords[:,0].int()== i).squeeze()
            batch_start = index_list[0]
            voxel_coords_batch= torch.index_select(voxel_coords, 0, index_list).cuda()
            inbox_index_bool= ~(voxel_coords_batch[:,0].int()== i)
            density_index_list=[]
            density_index_num_list=[]
            for i in range(1,20):
                for j in range(1,20):
                    index_bool = (voxel_coords_batch[:,1]< voxel_range[1]*i/19) & \
                                (voxel_coords_batch[:,1]>= voxel_range[1]*(i-1)/19) & \
                                    (voxel_coords_batch[:,2]>= voxel_range[0]*(j-1)/19) & \
                                    (voxel_coords_batch[:,2]< voxel_range[0]*j/19)
                    if index_bool.sum() > 0 :
                        density_index_num_list.append(index_bool.sum().item())
                        density_index_list.append(index_bool)
            # print(voxel_coords_batch)
            density_index_num_list = torch.tensor(density_index_num_list,dtype=int).cuda()
            # density_index_num_list = density_index_num_list*density_index_num_list
            density_index_num_list = density_index_num_list**(0.5)
            density_index_num_list = density_index_num_list / density_index_num_list.sum()
            density_index_num_list = (density_index_num_list*voxel_coords_batch.size()[0]*self.model_cfg["VOXEL_PERCENT"]).type(torch.int)
            for i in range(coor_box.size()[0]):
                in_bool= self.in_box_voxel(coor_box[i],voxel_coords_batch[:,1:])
                inbox_index_bool = inbox_index_bool | in_bool
            for index ,density_index_bool in enumerate(density_index_list):
                inbox_index_list = torch.nonzero(density_index_bool&inbox_index_bool).squeeze(dim=1)
                inbox_num = (density_index_bool&inbox_index_bool).sum().item()
                if density_index_num_list[index] == 0:
                    continue
                if density_index_num_list[index]> inbox_num:
                    outbox_index_list = torch.nonzero(density_index_bool&(~inbox_index_bool)).squeeze(dim=1)
                    point_num = density_index_num_list[index] - inbox_num
                    if point_num >outbox_index_list.numel():
                        point_num = outbox_index_list.numel()
                    out_keep_index = random.sample(outbox_index_list.tolist(), int(point_num))
                    out_keep_index = torch.tensor(out_keep_index, dtype=torch.int).cuda()
                    keep_index_list=torch.cat([inbox_index_list,out_keep_index],dim=0)
                else:
                    keep_index_list=random.sample(inbox_index_list.tolist(),\
                        int(density_index_num_list[index]))
                    keep_index_list = torch.tensor(keep_index_list, dtype=torch.int).cuda()
                keep_index_list = keep_index_list+batch_start
                keep_index_list_total.append(keep_index_list)
        keep_index_list_total=torch.cat(keep_index_list_total,dim=0)
        voxel_coords= torch.index_select(voxel_coords, 0, keep_index_list_total)
        points_mean= torch.index_select(points_mean, 0, keep_index_list_total)
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        return batch_dict

