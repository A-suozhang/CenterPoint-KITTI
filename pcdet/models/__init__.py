from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset, times):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset,times=times,
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'ret_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        # print(batch_dict['frame_id'])
        # exit()
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()

        # ===== apply reg logic here =====
        # if 'predictor_loss' in ret_dict.keys():
        #     loss += ret_dict['predictor_loss']

        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict,ret_dict)

    return model_func
