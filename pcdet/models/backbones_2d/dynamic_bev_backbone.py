import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        self.heatmaps = None
        self.save_dict={}
        self.save_dict['none_sparse_ratio']=[]
        self.save_dict['not_in_mask_gt_ratio']=[]
        self.debug = False
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            self.blocks.append(nn.ModuleList())
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            self.blocks[idx].append(nn.Sequential(*cur_layers))
            for k in range(layer_nums[idx]):
                cur_layers=[
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
                self.blocks[idx].append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        none_sparse_ratio=[]
        not_in_mask_gt_ratio=[]
        x = spatial_features
        gt_heatmap = torch.sum(self.heatmaps[0],dim=1,keepdim=True)
        gt_heatmap = F.max_pool2d(gt_heatmap, kernel_size=2)
        spatial_mask = (spatial_features.sum(dim=1,keepdim=True)!=0).int()
        num_gt_point = (gt_heatmap!=0).sum().item()
        num_not_in_mask_point = num_gt_point-((gt_heatmap*spatial_mask)!=0).sum().item()
        none_sparse_ratio.append(float(spatial_mask.sum().item())/float(spatial_mask.numel()))
        not_in_mask_gt_ratio.append(num_not_in_mask_point/num_gt_point)
        for i in range(len(self.blocks)):
            #import ipdb; ipdb.set_trace()
            for j,block in enumerate(self.blocks[i]):
                if False:#j==(len(self.blocks[i])-1): #odd(wo end), every(wo end), middle, begin
                    x = block(x)
                    x = spatial_mask*x
                else:
                    for layer in block:
                        x = layer(x)
                        if isinstance(layer,nn.Conv2d):
                            sparse_i = (x.sum(dim=1,keepdim=True)!=0).int()
                    if j%2==0:
                        # x = block(x)
                        x = spatial_mask*x
                        sparse_i = spatial_mask
                    if self.debug: # re-sparse 
                        x = x*sparse_i
                        num_not_in_mask_point = num_gt_point-((gt_heatmap*sparse_i)!=0).sum().item()
                        none_sparse_ratio.append(float(sparse_i.sum().item())/float(sparse_i.numel()))
                        not_in_mask_gt_ratio.append(num_not_in_mask_point/num_gt_point)
                        pass
                    

        
            self.save_dict['none_sparse_ratio'].append(none_sparse_ratio)
            self.save_dict['not_in_mask_gt_ratio'].append(not_in_mask_gt_ratio)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
