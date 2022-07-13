import torch

from .vfe_template import VFETemplate

import math
# 1 矩形平行于坐标轴
def isInMatrix(x1,y1,x2,y2,x,y):
    if x<=x1 or x>=x2 or y<=y1 or y>=y2: return False
    return True

# 2 矩形为一般矩形，可旋转
def isInSide(x1,y1,x2,y2,x3,y3,x4,y4,x,y):
    if y1==y2: return isInMatrix(x1,y1,x4,y4,x,y)
    l,k,s=y4-y3,x4-x3,math.sqrt((x4-x3)**2+(y4-y3)**2)
    cos,sin=l/s,k/s
    x1r,y1r=x1*cos+y1*sin,y1*cos-x1*sin
    x4r,y4r=x4*cos+y4*sin,y4*cos-x4*sin
    xr,yr=x*cos+y*sin,y*cos-x*sin
    return isInMatrix(x1r,y1r,x4r,y4r,xr,yr)
class DropVoxelVFE(VFETemplate):
    # The pre-prcocess-based Drop Voxel should be defined here
    # TODO: input interface for point-level feature.
    # INFO: however, there would be many ops within the 3d backbone inference process, how to feed in?
    # write another OP maybe?
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        # type xxx.

    def get_output_feature_dim(self):
        return self.num_point_features

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
        import ipdb; ipdb.set_trace()
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        # ==== EXP: Oracle-0 of the DropVoxel: Randomly drop voxel ====

        # ==== EXP: Oracle-1 of the DropVoxel: Randomly drop voxel outside the bbox====
        # TODO:
        #   - config of different dropping techniques
        #   - find the voxels outside the bbox(create an idx list)
        #   - random choice of the voxel-not-in-box
        #

        # pc_range = torch.tensor(self.point_cloud_range)
        # voxel_size = torch.tensor(self.target_cfg.VOXEL_SIZE)
        # coor_x = (x - pc_range[0]) / voxel_size[0] 
        # coor_y = (y - pc_range[1]) / voxel_size[1] 

        # voxel_coods = batch_dict['voxel_coords']
        # gt_boxes = batch_dict["gt_boxes"]
        # print(gt_boxes.size())
        # print(points_mean.size())
        # print(points_mean[0])
        # print(voxel_coods.size())
        # print(voxel_coods[0])
        # print(gt_boxes[0])
        # exit()
        return batch_dict

