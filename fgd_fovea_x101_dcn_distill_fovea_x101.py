dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='CocoDataset',
        classes=('caption', 'table', 'figure'),
        ann_file=
        '../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/train.json',
        img_prefix='../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('caption', 'table', 'figure'),
        ann_file='../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/val.json',
        img_prefix='../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('caption', 'table', 'figure'),
        ann_file=
        '../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/test.json',
        img_prefix='../mmdetection/data/Truong_Hai/dataset/UIT-DODV-Ext/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 5e-06
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained=
    '../mmdetection/data/Truong_Hai/work_dirs/fovea_align_x101_64x4d_fpn_gn-head_mstrain_640-800_4x4_2x_dodv_ext/best_bbox_mAP_epoch_19.pth',
    init_student=True,
    distill_cfg=[
        dict(
            student_module='neck.fpn_convs.4.conv',
            teacher_module='neck.fpn_convs.4.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_4',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.3.conv',
            teacher_module='neck.fpn_convs.3.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_3',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.2.conv',
            teacher_module='neck.fpn_convs.2.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_2',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.1.conv',
            teacher_module='neck.fpn_convs.1.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_1',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.0.conv',
            teacher_module='neck.fpn_convs.0.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_0',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ])
    ])
student_cfg = '../mmdetection/data/Truong_Hai/fovea_pth/fovea_align_x101_64x4d_fpn_gn-head_mstrain_640-800_4x4_2x_dodv_ext/fovea_align_x101_64x4d_fpn_gn-head_mstrain_640-800_4x4_1x_dodv_ext.py'
teacher_cfg = '../mmdetection/data/Truong_Hai/work_dirs/fovea_align_x101_64x4d_fpn_gn-head_mstrain_640-800_4x4_2x_dodv_ext/fovea_align_x101_64x4d_dcn_fpn_gn-head_mstrain_640-800_4x4_2x_dodv_ext.py'
work_dir = '../mmdetection/data/Truong_Hai/work_dirs/fgd_fovea_x101_dcn_distill_fovea_x101'
auto_resume = False
gpu_ids = range(0, 1)
