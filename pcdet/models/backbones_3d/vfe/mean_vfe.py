import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

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
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()
        #以下为drop voxel实验代码
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
