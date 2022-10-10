from .detector3d_template import Detector3DTemplate
import torch
from .gt_heatmap import get_nearby_centroids_offset
import torch.nn.functional as F

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset,times):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset,times = times)
        self.module_list = self.build_networks()
        self.i = 0
        self.debug = model_cfg['DEBUG'] if 'debug' in model_cfg.keys() else None
        self.radius = model_cfg['BACKBONE_3D']['RADIUS'] if 'RADIUS' in model_cfg['BACKBONE_3D'].keys() else None
        self.use_predictor = model_cfg['BACKBONE_3D']['use_predictor'] if 'use_predictor' in model_cfg['BACKBONE_3D'].keys() else False
        self.custom_gt_heatmap = model_cfg['BACKBONE_3D']['custom_gt_heatmap'] if 'custom_gt_heatmap' in model_cfg['BACKBONE_3D'].keys() else False
        # self.use_predictor = model_cfg['BACKBONE_3D']['use_predictor']

        self.save_dict = {}
        self.save_dict['train_loss'] = []

    def forward(self, batch_dict):
        # introducing new predictor module* 
        # used when applying predictor
        # need to use head to gen the heatmap, as guide of predictor training

        if self.use_predictor:
            if self.custom_gt_heatmap:
                if self.custom_gt_heatmap == 'nearby_centroid':
                    offsets = get_nearby_centroids_offset(batch_dict,radius=self.radius,dataset_cfg=self.dataset,model_cfg=self.model_cfg)
                    gt_boxes_corrected = batch_dict['gt_boxes'].clone()
                    gt_boxes_corrected[:,:,:2] += offsets
                    heatmaps = self.module_list[4].assign_targets(gt_boxes_corrected,radiu=self.radius)['heatmaps']

                    # save_d = {
                    #         'coords': batch_dict['voxel_coords'],
                    #         'gt_boxes':batch_dict['gt_boxes'],
                    #         'gt_boxes_corrected': gt_boxes_corrected,
                    #         'heatmap':heatmaps,
                    #         }
                    # torch.save(save_d,'debug_nearby_centroid_heatmap.pth')
                    # import ipdb; ipdb.set_trace()
                else:
                    raise NotImplementedError
            else:
                heatmaps = self.module_list[4].assign_targets(batch_dict['gt_boxes'],radiu=self.radius)['heatmaps']


            # gt_heatmap = torch.mean(heatmaps[0],dim=1)
            # gt_heatmap = F.avg_pool2d(gt_heatmap, kernel_size=2)   
            # gt_heatmap1 = torch.mean(heatmaps1[0],dim=1)
            # gt_heatmap1 = F.avg_pool2d(gt_heatmap1, kernel_size=2) 
            # save_dict = {}
            # save_dict['gt_heatmap'] = gt_heatmap[0]
            # save_dict['gt_heatmap1'] = gt_heatmap1[0]
            # torch.save(save_dict, f"./visualization/voxels/voxels_GT.pth")
            # exit() 
            self.module_list[1].gt_heatmap = heatmaps # feed into the 3d backbone
        if self.debug:
            heatmaps = self.module_list[4].assign_targets(batch_dict['gt_boxes'],radiu=None)['heatmaps']
            self.module_list[3].heatmaps = heatmaps
            self.module_list[3].debug = self.debug
        # ======== 
        # 0 - VFE
        # 1 - 3D Backbone
        # 2 - HeightCompression
        # 3 - 2D Backbone
        # 4 - Head
        # ========
        for idx_, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
        self.i +=1
        # print(batch_dict['spatial_features_2d'][0,0,0,0])
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            self.save_dict['train_loss'].append(loss.item())

            if hasattr(self.module_list[1],'predictor_loss'):
                predictor_loss = self.module_list[1].predictor_loss
                ret_dict = {
                'loss': loss,
                'predictor_loss': predictor_loss,
                }
            else:
                ret_dict = {
                'loss': loss,
                }
            # ret_dict = {
            #     'loss': loss,
            #     'predictor_loss': predictor_loss,
            # }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss()
            voxel_coords = batch_dict['voxel_coords']
            gt_boxes = batch_dict['gt_boxes']
            d = {}
            d['gt_boxes'] = gt_boxes
            d['voxel_coords'] = voxel_coords
            d['pred_dicts'] = pred_dicts
            torch.save(d, f"./visualization/pred_dict_{self.model_cfg.VFE['NAME']}{self.model_cfg.VFE['VOXEL_PERCENT']}.pth")
            # import ipdb; ipdb.set_trace()
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        # TODO: new reg term, here
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        # predictor loss
        if self.use_predictor:
            predictor_loss = self.module_list[1].predictor_loss
        else:
            predictor_loss = 0

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'predictor_loss': predictor_loss,
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
