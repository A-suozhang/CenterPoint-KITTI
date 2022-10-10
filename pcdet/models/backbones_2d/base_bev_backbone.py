import numpy as np
import torch
import torch.nn as nn


class MaskedBatchNorm2d(nn.Module):

    def __init__(self, C, eps=1e-3, momentum=0.1, affine=False):
        super().__init__()
        # self.bn = nn.BatchNorm2d(C, momentum=1., eps=eps, affine=affine)   # DEBUG_ONLY
        self.momentum = momentum
        self.eps = eps
        self.running_mean = 0.
        self.running_var = 1.
        self.training = True

    def forward(self, x):
        # return x # DEBUG_ONLY

        B,C,W,H = x.shape
        x1=x.sum(1)
        nonzero_idxs = []
        num_nonzeros = []
        for _ in range(B):
            nonzero_idx_cur_batch = x1[_].nonzero()
            num_nonzero = nonzero_idx_cur_batch.shape[0]
            nonzero_idxs.append(nonzero_idx_cur_batch)
            num_nonzeros.append(num_nonzero)
        # print('nonzero-rate', max(num_nonzeros)/(W*H))
        idx_for_gather = torch.zeros([B,max(num_nonzeros),2],device=x.device)
        for _ in range(B):
            idx_for_gather[_,:num_nonzeros[_],:] = nonzero_idxs[_]
        # since different batch has different nonzero elments
        # zero-padded, gathered then replaced with zero(since the upper left element is always empty)
        # Inputs: [B,C,W,J]
        # idxs: [B,K,2]
        # gathered: [B,C,K]
        idx_for_gather = idx_for_gather.long()
        batch_enum = torch.arange(B).unsqueeze(1)
        gathered = x[batch_enum,:,idx_for_gather[:,:,0],idx_for_gather[:,:,1]]
        gathered = gathered.permute([0,2,1]).unsqueeze(-1).contiguous()  # [B,C,K]
        # save_d = {}
        # save_d['pre-bn'] = gathered
        # gathered_bn = self.bn(gathered)

        mean_ = gathered.mean(dim=(0,2,3),keepdim=True)
        if not self.training:
            var_ = self.running_var
        else:
            var_ = ((gathered - mean_)**2).mean(dim=(0,2,3), keepdim=True)
            self.running_var = (1.-self.momentum)*self.running_var + self.momentum*var_

        # DEBUG: when substracting mean, it wont work
        # variance only
        gathered_bn = (gathered) / torch.sqrt(var_ + self.eps)
        # gathered_bn = self.bn(gathered)


        # print((mean_.reshape([C]) - self.bn.running_mean).abs().max())
        # print((var_.reshape([C]) - self.bn.running_var).abs().max())
        # print((gathered_bn_ - gathered_bn).abs().max())

        # save_d['post-bn'] = gathered_bn

        # torch.save(save_d, './visual_utils/probe-bn-temp.pth')
        # import ipdb; ipdb.set_trace()
        x[batch_enum,:,idx_for_gather[:,:,0],idx_for_gather[:,:,1]] = gathered_bn.squeeze(-1).permute([0,2,1]).contiguous()
        # x[:,:,0,0] = x[:,:,0,0]*0.   # mask the upper left

        return x

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.save_dict = {}

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

        # DEBUG_ONLY
        # self.masked_batchnorm = True
        # bn_type = MaskedBatchNorm2d

        if self.model_cfg.get('MASKED_BATCHNORM', None) is not None:
            self.masked_batchnorm = True
            bn_type = MaskedBatchNorm2d
        else:
            self.masked_batchnorm = False
            bn_type = nn.BatchNorm2d

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                bn_type(num_filters[idx], eps=1e-3, momentum=0.01, affine=False),  # DEBUG_ONLY: setting the affine=false
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    bn_type(num_filters[idx], eps=1e-3, momentum=0.01, affine=False),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        bn_type(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
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
                        bn_type(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                bn_type(c_in, eps=1e-3, momentum=0.01),
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
        x = spatial_features
        for i in range(len(self.blocks)):

            EXPORT_ACT=False  # DEBUG_ONLY, false as default
            if not EXPORT_ACT:
                x = self.blocks[i](x)   # COMMENT OUT for DEBUG_ONLY
            else:
                # DEBUG_ONLY: Export the activations
                conv_id = [1,4,7,10,13,16] # 6 convs
                bn_id = [2,5,8,11,14,17]
                for _ in range(len(conv_id)):
                    self.save_dict['in_feat_{}'.format(_)] = []
                    self.save_dict['out_feat_{}'.format(_)] = []
                    self.save_dict['conv_weight_{}'.format(_)] = []

                save_d = {}
                for m,n in enumerate(self.blocks[i]):
                    # print(m,n)
                    if m in conv_id:
                        # save_d['pre-conv']=x
                        self.save_dict['in_feat_{}'.format(conv_id.index(m))].append(x.detach().cpu())
                        x = n(x)
                        # save_d['post-conv']=x
                        self.save_dict['out_feat_{}'.format(conv_id.index(m))].append(x.detach().cpu())
                        self.save_dict['conv_weight_{}'.format(conv_id.index(m))].append(self.blocks[i][m].weight.detach().cpu())
                    # elif m in bn_id:
                        # # TODO: manual batchnorm2d, with skipping zero
                        # # save_d['pre-bn']=x
                        # if self.masked_batchnorm:
                            # x = masked_batchnorm_forward(x,m)
                        # else:
                            # x = n(x)
                        # save_d['post-bn']=x
                        # torch.save(save_d,'./visual_utils/probe-bn.pth')
                    else:
                        x = n(x)
                torch.save(self.save_dict,'./visual_utils/2d_conv_features.pth')
                import ipdb; ipdb.set_trace()

            for s in self.save_dict.keys():
                if 'feat' in s:
                    for _ in range(len(self.save_dict[s])):
                        feat = self.save_dict[s][_].sum(1)  # [bs,c,w,h]
                    sparse_rate = (feat==0).sum()/feat.nelement()
                    print('feat {},sparse_rate:{:.3f}'.format(s, sparse_rate))

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
