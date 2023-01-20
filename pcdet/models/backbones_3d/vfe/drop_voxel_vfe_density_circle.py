import torch

from .vfe_template import VFETemplate
import random
import math
import numpy as np
import copy

class DropVoxelDensityOriginVFE(VFETemplate):
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
        voxels_cood_y = voxels_cood[:,2]*torch.sin(gtbox[3]) \
        + voxels_cood[:,1]*torch.cos(gtbox[3])
        voxels_cood_x = -voxels_cood[:,1]*torch.sin(gtbox[3]) \
        + voxels_cood[:,2]*torch.cos(gtbox[3])
        voxels_cood[:,1] = voxels_cood_y
        voxels_cood[:,2] = voxels_cood_x
        # import ipdb; ipdb.set_trace()
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
        voxel_coods = batch_dict['voxel_coords']
        pc_range = torch.tensor(point_cloud_range)
        voxel_size = torch.tensor(voxel_size)
        voxel_range = ((pc_range[3:]-pc_range[0:3])/voxel_size).cuda()
        gt_boxes = batch_dict['gt_boxes']
        origin_x = (- pc_range[0] / voxel_size[0]).cuda()
        origin_y = (- pc_range[1] / voxel_size[1]).cuda()
        maxdistance = (voxel_range[1]**2 +voxel_range[0]**2).cuda()
        # rotation = (-(gt_boxes[:,:,6]+np.pi/2))%np.pi
        keep_index_list_total =[]
        for i in range(batch_dict['batch_size']):
            label = gt_boxes[i,:,7]
            index_list = torch.nonzero(label.int()!= 0).squeeze().cuda()
        
            index_list = torch.nonzero(voxel_coods[:,0].int()== i).squeeze().cuda()
            batch_start = index_list[0]
            voxel_coods_batch= torch.index_select(voxel_coods, 0, index_list).cuda()
            density_index_list=[]
            density_index_num_list=[]
            distance_list = []
            distance_batch = ((voxel_coods_batch[:,2]-origin_x)**2 +(voxel_coods_batch[:,1]-origin_y)**2).cuda()
            # print("123")
            for i in range(1,40):
                    index_bool = (distance_batch > ((maxdistance*(i-1))/39)) \
                        & (distance_batch< ((maxdistance*i)/39))
                    if index_bool.sum() > 0 :
                        density_index_num_list.append(index_bool.sum().item())
                        density_index_list.append(index_bool.cuda())
            density_index_num_list = (torch.tensor(density_index_num_list,dtype=int)).cuda()
            density_index_num_list = density_index_num_list**(0.5)
            density_index_num_list = density_index_num_list / density_index_num_list.sum()
            density_index_num_list = (density_index_num_list*voxel_coods_batch.size()[0]*self.model_cfg["VOXEL_PERCENT"]).type(torch.int)
            for index ,density_index_bool in enumerate(density_index_list):
                inbox_index_list = torch.nonzero(density_index_bool).squeeze(dim=1).cuda()
                if inbox_index_list.numel() > density_index_num_list[index]:
                    keep_index_list=random.sample(inbox_index_list.tolist(),\
                            int(density_index_num_list[index]))
                else:
                    keep_index_list = inbox_index_list.tolist()
                keep_index_list = torch.tensor(keep_index_list, dtype=torch.int).cuda()
                keep_index_list = keep_index_list+batch_start
                keep_index_list_total.append(keep_index_list)
        keep_index_list_total=torch.cat(keep_index_list_total,dim=0).cuda()
        voxel_coods= torch.index_select(voxel_coods, 0, keep_index_list_total).cuda()
        points_mean= torch.index_select(points_mean, 0, keep_index_list_total).cuda()
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coods.contiguous()
        return batch_dict

