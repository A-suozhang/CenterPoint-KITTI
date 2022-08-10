from .detector3d_template import Detector3DTemplate
import torch

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset,times):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset,times = times)
        self.module_list = self.build_networks()
        self.use_predictor = model_cfg['BACKBONE_3D']['use_predictor']

    def forward(self, batch_dict):
        # introducing new predictor module* 
        # used when applying predictor
        # need to use head to gen the heatmap, as guide of predictor training

        if self.use_predictor:
            heatmaps = self.module_list[4].assign_targets(batch_dict['gt_boxes'])['heatmaps']
            self.module_list[1].gt_heatmap = heatmaps # feed into the 3d backbone

        # ======== 
        # 0 - VFE
        # 1 - 3D Backbone
        # 2 - HeightCompression
        # 3 - 2D Backbone
        # 4 - Head
        # ========
        for idx_, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            if hasattr(self.module_list[1],'predictor_loss'):
                predictor_loss = self.module_list[1].predictor_loss
            else:
                predictor_loss = 0.

            ret_dict = {
                'loss': loss,
                'predictor_loss': predictor_loss,
            }
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
            import ipdb; ipdb.set_trace()
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
