voxel_size = .01
n_points = 100000
plugin = True
plugin_dir = "projects_incre/incre/"
mode = 'alpha'

if mode == 'count':
    inc_class_names = ('chair','door','garbagebin','cabinet','table','window','picture','desk','sofa',
                                'sink','bed','bookshelf','curtain','counter','toilet','refrigerator','showercurtrain','bathtub')
else:
    inc_class_names = None


model = dict(
    type='MinkSingleStage3DDetector_INC',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, max_channels=128, depth=34, norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='Base_stage',
        class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin'),
        inc_class_names = inc_class_names,
        mode = mode,
        anno_type = '9_1_increment',
        in_channels=128,
        n_reg_outs=6,
        # base_n_class=9, # 18-9 !!!!
        n_classes=18, 
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))


dataset_type = 'ScanNetDataset_INC'
data_root = '../all_data/scannet_v2/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D_INC', 
         with_bbox_3d=True, 
         with_label_3d=True, 
         with_feats=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    # we do not sample 100k points for scannet, as very few scenes have
    # significantly more then 100k points. so we sample 33 to 100% of them
    dict(type='PointSample', num_points=.33),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, .02],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D_INC', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','gt_feats'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            # we do not sample 100k points for scannet, as very few scenes have
            # significantly more then 100k points. so it doesn't affect inference
            # time and we ca accept all points
            # dict(type='PointSample', num_points=n_points),
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

ann_root = '../all_data/scannet_v2/few_shot_increment/'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=15,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=ann_root + 'scannet_infos_train_few_shot_incremental_' + mode +'.pkl',
            pesudo_ann_file='../all_data/scannet_v2/pesudo_pkl/scannet_infos_train_object_pesudo_9_feat_quality_' + mode + '.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth',
            increment_pkl = True,
            # if increment_pkl = old, just make sure the 'anno_type' first number is right
            # if increment_pkl = old, make sure 'anno_type' all match
            increament_type = 'old', 
            anno_type = '9_1_increment',), # must_same_with_head
            ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        anno_type = '9_1_increment',
        mode = mode),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        anno_type = '9_1_increment',
        mode = mode))


optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=10)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]