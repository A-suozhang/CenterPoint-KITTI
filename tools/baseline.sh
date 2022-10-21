export OMP_NUM_THREADS=1
# python train.py --cfg_file cfgs/kitti_models/centerpoint-predictor-train.yaml  --extra_tag $1 --gpu $2 --pretrained_model ../pcdet/output/home/nfs_data/lupu/CenterPoint-KITTI/tools/cfgs/kitti_models/centerpoint/default/ckpt/checkpoint_epoch_80.pth --train_mode 1
python train.py --cfg_file cfgs/kitti_models/bak/centerpoint.yaml  --extra_tag $1 --gpu $2 --train_mode 0
#python train.py --cfg_file cfgs/kitti_models/bak/masked_bn.yaml  --extra_tag $1 --gpu $2 --train_mode 0
