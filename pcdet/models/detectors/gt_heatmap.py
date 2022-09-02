import torch 
import numpy as np
import torch.nn as nn
import copy
from functools import partial
from six.moves import map, zip


# def multi_apply(func, *args, **kwargs):
    # pfunc = partial(func, **kwargs) if kwargs else func
    # map_results = map(pfunc, *args)
    # return tuple(map(list, zip(*map_results)))

# def gaussian_2d(shape, sigma=1):
    # """Generate gaussian map.

    # Args:
        # shape (list[int]): Shape of the map.
        # sigma (float): Sigma to generate gaussian map.
            # Defaults to 1.

    # Returns:
        # np.ndarray: Generated gaussian map.
    # """
    # m, n = [(ss - 1.) / 2. for ss in shape]
    # y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # return h

# def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    # """Get gaussian masked heatmap.

    # Args:
        # heatmap (torch.Tensor): Heatmap to be masked.
        # center (torch.Tensor): Center coord of the heatmap.
        # radius (int): Radius of gausian.
        # K (int): Multiple of masked_gaussian. Defaults to 1.

    # Returns:
        # torch.Tensor: Masked heatmap.
    # """
    # diameter = 2 * radius + 1
    # gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    # x, y = int(center[0]), int(center[1])

    # height, width = heatmap.shape[0:2]

    # left, right = min(x, radius), min(width - x, radius + 1)
    # top, bottom = min(y, radius), min(height - y, radius + 1)

    # masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    # masked_gaussian = torch.from_numpy(
        # gaussian[radius - top:radius + bottom,
                 # radius - left:radius + right]).to(heatmap.device,
                                                   # torch.float32)
    # if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        # torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # return heatmap


# def get_targets_single(dataset,model_cfg, voxels_coord, radius):
    # device = voxels_coord.device
    # """gt_bboxes_3d = torch.cat(
        # (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
        # dim=1).to(device)
    # """

    # max_objs = 100
    # grid_size = torch.tensor(dataset.grid_size)
    # feature_map_size = grid_size[:2] // model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.OUT_SIZE_FACTOR

    # task_voxels = [voxels_coord]
    # # task_classes = [gt_labels_3d]

    # draw_gaussian = draw_heatmap_gaussian
    # heatmaps =  []

    # for idx in range(1):
        # heatmap = voxels_coord.new_zeros(
            # (1, feature_map_size[1],
                # feature_map_size[0]))
        # num_objs = task_voxels[idx].shape[0]

        # for k in range(num_objs):
            # if radius is None:
                # raise ValueError('A very specific bad thing happened.')
            # x, y, z = task_voxels[idx][k][3], task_voxels[idx][k][
                # 2], task_voxels[idx][k][1]

            # coor_x = x  / model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.OUT_SIZE_FACTOR
            # coor_y = y / model_cfg.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.OUT_SIZE_FACTOR

            # center = torch.tensor([coor_x, coor_y],
                                    # dtype=torch.float32,
                                    # device=device)
            # center_int = center.to(torch.int32)

            # # throw out not in range objects to avoid out of array
            # # area when creating the heatmap
            # if not (0 <= center_int[0] < feature_map_size[0]
                    # and 0 <= center_int[1] < feature_map_size[1]):
                # continue

            # draw_gaussian(heatmap[0], center_int, radius)


        # heatmaps.append(heatmap)
    # return heatmaps

# def assign_targets_voxels(voxels,dataset_cfg,model_cfg,radiu = None):
    # radius=[]
    # dataset_cfgs = []
    # model_cfgs = []
    # for i in range(len(voxels)):
        # radius.append(radiu)
        # dataset_cfgs.append(dataset_cfg)
        # model_cfgs.append(model_cfg)
    # heatmaps = multi_apply(
         # get_targets_single, dataset_cfgs,model_cfgs,voxels,radius)
    # heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
    # heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
    # return heatmaps


# def in_box_voxel(gtbox,voxels_coord):
#     voxels_coord = copy.deepcopy(voxels_coord)
#     voxels_coord[:,]=voxels_coord[:,] - gtbox[0:3] #z,y,x
#     voxels_coord_y = voxels_coord[:,2]*torch.sin(gtbox[3]) + voxels_coord[:,1]*torch.cos(gtbox[3])
#     voxels_coord_x = -voxels_coord[:,1]*torch.sin(gtbox[3]) + voxels_coord[:,2]*torch.cos(gtbox[3])
#     voxels_coord[:,1] = voxels_coord_y
#     voxels_coord[:,2] = voxels_coord_x
#     # import ipdb; ipdb.set_trace()
#     in_bool =   (voxels_coord[:,1] <= (gtbox[5]+1)/2) & (voxels_coord[:,1] >= -(gtbox[5]+1)/2) \
#     &(voxels_coord[:,2] <= (gtbox[6]+1)/2) & (voxels_coord[:,2] >= -(gtbox[6]+1)/2)\
#     &(voxels_coord[:,0]<=(gtbox[4]+1)/2)&(voxels_coord[:,0]>=-(gtbox[4]+1)/2)
#     return in_bool

# def batch_in_boxes_voxeles(batch_dict,dataset_cfg):
#     point_cloud_range = dataset_cfg.point_cloud_range
#     voxel_size = dataset_cfg.voxel_size
#     voxel_coords = batch_dict['voxel_coords']
#     pc_range = torch.tensor(point_cloud_range)
#     voxel_size = torch.tensor(voxel_size)
#     gt_boxes = batch_dict['gt_boxes']
#     coor_centerx = (gt_boxes[:,:,0] - pc_range[0]) / voxel_size[0] 
#     coor_centery = (gt_boxes[:,:,1] - pc_range[1]) / voxel_size[1] 
#     coor_centerz = (gt_boxes[:,:,2] - pc_range[2]) / voxel_size[2] 
#     coor_x = (gt_boxes[:,:,3]) / voxel_size[0] 
#     coor_y = (gt_boxes[:,:,4]) / voxel_size[1] 
#     coor_z = (gt_boxes[:,:,5]) / voxel_size[2] 
#     # rotation = (-(gt_boxes[:,:,6]+np.pi/2))%np.pi
#     rotation = gt_boxes[:,:,6]
#     keep_index_list_total =[]
#     voxel_batch_list = []
#     for i in range(batch_dict['batch_size']):
#         label = gt_boxes[i,:,7]
#         index_list = torch.nonzero(label.int()!= 0).squeeze()
#         coor_centerx_batch = torch.index_select(coor_centerx[i],0,index_list)
#         coor_centery_batch = torch.index_select(coor_centery[i],0,index_list)
#         coor_centerz_batch = torch.index_select(coor_centerz[i],0,index_list)
#         coor_x_batch = torch.index_select(coor_x[i],0,index_list)
#         coor_y_batch = torch.index_select(coor_y[i],0,index_list)
#         coor_z_batch = torch.index_select(coor_z[i],0,index_list)
#         rotation_batch = - torch.index_select(rotation[i],0,index_list)
#         coor_box =torch.stack((coor_centerz_batch,coor_centery_batch,coor_centerx_batch,rotation_batch,
#                                                                 coor_z_batch,coor_y_batch,coor_x_batch),dim=1)
#         index_list = torch.nonzero(voxel_coords[:,0].int()== i).squeeze()
#         batch_start = index_list[0]
#         voxel_coords_batch= torch.index_select(voxel_coords, 0, index_list)
#         # outbox_index_list=[]
#         # print(voxel_coords_batch)
#         coor_box1 = torch.unsqueeze(coor_box, dim=1)
#         voxel_coords_batch1 = torch.unsqueeze(voxel_coords_batch[:,1:],dim=0)
#         voxel_coords_batch1 = voxel_coords_batch1 - coor_box1[:,:,0:3]
#         voxels_coord_y = voxel_coords_batch1[:,:,2]*torch.sin(coor_box1[:,:,3]) + voxel_coords_batch1[:,:,1]*torch.cos(coor_box1[:,:,3])
#         voxels_coord_x = -voxel_coords_batch1[:,:,1]*torch.sin(coor_box1[:,:,3]) + voxel_coords_batch1[:,:,2]*torch.cos(coor_box1[:,:,3])
#         voxel_coords_batch1[:,:,1]=voxels_coord_y
#         voxel_coords_batch1[:,:,2] = voxels_coord_x
#         in_bool1 =   (voxel_coords_batch1[:,:,1] <= (coor_box1[:,:,5]+1)/2) & (voxel_coords_batch1[:,:,1] >= -(coor_box1[:,:,5]+1)/2) \
#         &(voxel_coords_batch1[:,:,2] <= (coor_box1[:,:,6]+1)/2) & (voxel_coords_batch1[:,:,2] >= -(coor_box1[:,:,6]+1)/2)\
#         &(voxel_coords_batch1[:,:,0]<=(coor_box1[:,:,4]+1)/2)&(voxel_coords_batch1[:,:,0]>=-(coor_box1[:,:,4]+1)/2)
#         in_bool1 = torch.any(in_bool1, 0)
#         # for i in range(coor_box.size()[0]):
#         #     # print(coor_box[i])
#         #     in_bool= in_box_voxel(coor_box[i],voxel_coords_batch[:,1:])
#         #     inbox_index_bool = inbox_index_bool | in_bool
#         keep_index_list = torch.nonzero(in_bool1).squeeze(dim=1)
#         keep_index_list = keep_index_list+batch_start
#         # keep_index_list_total.append(keep_index_list)
#         voxel_coords_batch= torch.index_select(voxel_coords, 0, keep_index_list)
#         voxel_batch_list.append(voxel_coords_batch)
#     # batch_voxels = torch.stack(voxel_batch_list)
#     # batch_voxels = [torch.stack(batVo) for batVo in voxel_batch_list]
#     return voxel_batch_list

def get_nearby_centroids_offset(batch_dict,radius,dataset_cfg,model_cfg):

    # gen the gt-centroid offset(within voxel-domain)
    point_cloud_range = dataset_cfg.point_cloud_range    # [6,]
    voxel_size = dataset_cfg.voxel_size    # [3,]

    coords = batch_dict['voxel_coords']  # [N, 3]
    gt_boxes = batch_dict['gt_boxes']  # [bs, N-box, 8]
    # transform from [1600, 1400] -> [300, 200]
    coords_center = (gt_boxes[:,:,0:3] - torch.tensor(point_cloud_range[0:3], device=gt_boxes.device).reshape([1,1,-1])) / torch.tensor(voxel_size, device=gt_boxes.device).reshape([1,1,-1])  # [bs, N-box, 3]
    local_size = 20 # hyper-param to choose

    batch_ids = coords[:,0]
    batch_size = batch_ids.max().int().item()+1   # FIXME: dirty
    offsets = []
    for i_batch in range(batch_size):
        idxs_cur_batch = torch.nonzero(batch_ids==i_batch).squeeze(-1)   # [N]
        coords_cur_batch = torch.index_select(coords, 0, idxs_cur_batch)[:,[3,2]] # [N,2] only acquire x-y
        box_centers_cur_batch = coords_center[i_batch,:,0:2]  # [N-box, 2]
        dist = coords_cur_batch.unsqueeze(1) - box_centers_cur_batch.unsqueeze(0)  # [N,N-box,2]
        voxels_in_local = ((dist[:,:,0].abs()<local_size)&(dist[:,:,1].abs()<local_size)).int()   # [N, N-box] (0,1 mask)
        n_voxels_in_local = voxels_in_local.sum(0)  # [N-box]
        n_voxels_in_local = n_voxels_in_local+(n_voxels_in_local==0).int()  # ZeroMangement, incase no points within local-range, count as 1 voxel to prevent 0 division
        dist = dist*voxels_in_local.unsqueeze(-1) # [N, N-box, 2], filter the >local-range voxels
        dist = dist.sum(0)/n_voxels_in_local.unsqueeze(-1) # [N-box, 2], mean the coord within range
        dist = dist*torch.tensor(voxel_size[:2],device=gt_boxes.device).unsqueeze(0)  # revert to [W,H] via multipllying voxel-size

        # INFO: save the offsets
        # box_centers_cur_batch_corrected = box_centers_cur_batch + dist
        # save_d = {
        #         'coords': coords_cur_batch,
        #         'box-center': box_centers_cur_batch,
        #         'box-center-corrected': box_centers_cur_batch_corrected,
        #         }
        # torch.save(save_d, 'debug-for-nearby-centroids.pth')
        # import ipdb; ipdb.set_trace()
        offsets.append(dist)
    offsets = torch.stack(offsets, dim=0)  # [bs. N-box, 2]
    # CHECK: the empty [0,0] box-center, should not be offseted*
    # currently for a small local-range, naturally there would be no offsets
    return offsets

    # voxels = batch_in_boxes_voxeles(batch_dict,dataset_cfg)
    # heatmaps = assign_targets_voxels(voxels,dataset_cfg,model_cfg,radius)
    # return heatmaps
