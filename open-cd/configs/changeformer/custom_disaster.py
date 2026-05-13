
_base_ = ['../_base_/datasets/disaster_hetero_cd.py', '../_base_/default_runtime.py']
custom_imports = dict(imports=[
    'opencd.models.backbones.pseudo_siam', 
    'opencd.datasets.transforms.hetero_loading',
    'opencd.datasets.disaster_cd'
], allow_failed_imports=False)
model = dict(
    type='mmseg.EncoderDecoder',
    data_preprocessor=dict(
        type='mmseg.SegDataPreProcessor', 
        mean=[123.6, 116.2, 103.5, 127.5], std=[58.3, 57.1, 57.3, 57.1], 
        bgr_to_rgb=False, pad_val=0, seg_pad_val=0, size_divisor=32),
    backbone=dict(
        type='mmseg.PseudoSiamChangeFormer', 
        in_channels_eo=3, in_channels_sar=1, 
        backbone_cfg=dict(
            type='mmseg.MixVisionTransformer', 
            in_channels=3, embed_dims=64, num_stages=4, num_layers=[3, 4, 18, 3], 
            num_heads=[1, 2, 5, 8], patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1], 
            out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)),
    decode_head=dict(
        type='mmseg.SegformerHead', 
        in_channels=[128, 256, 640, 1024], in_index=[0, 1, 2, 3], channels=256, num_classes=2, 
        loss_decode=[
            dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, loss_name='loss_ce'),
            dict(type='mmseg.DiceLoss', loss_weight=1.0, loss_name='loss_dice')
        ]),
    train_cfg=dict(), test_cfg=dict(mode='whole'))
optim_wrapper = dict(type='mmengine.OptimWrapper', optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01))
param_scheduler = [dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500), dict(type='CosineAnnealingLR', begin=500, by_epoch=False, T_max=4500, eta_min=0.0)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=5000, val_interval=5000000000000000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500000000, max_keep_ckpts=3, save_best='mIoU'), logger=dict(type='LoggerHook', interval=500000000, log_metric_by_epoch=False), visualization=dict(type='HeteroVisHook', draw=True, interval=1))
