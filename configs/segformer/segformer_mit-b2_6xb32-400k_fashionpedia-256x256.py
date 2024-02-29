_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/fashionpedia_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_400k_own.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512], 
        num_classes=28
    ),
)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.0000001),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=12000),
    dict(
        type='PolyLR',
        power=0.95,
        begin=12000,
        end=400000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=32, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader

# overrride the default schedule
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')