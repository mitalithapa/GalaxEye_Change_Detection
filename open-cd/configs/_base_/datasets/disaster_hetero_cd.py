
dataset_type = 'DisasterCDDataset'
data_root = '/content/dataset'
crop_size = (256, 256)
pipeline = [
    dict(type='LoadHeteroImagesFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='BinarizeLabels'),
    dict(type='MultiImgRandomRotate', prob=0.5, degree=180),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='LoadHeteroImagesFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='BinarizeLabels'),
    dict(type='MultiImgPackSegInputs')
]
train_dataloader = dict(
    batch_size=8, num_workers=2, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, img_suffix='.tif', seg_map_suffix='.tif',
        data_prefix=dict(img_path_from='train/pre-event', img_path_to='train/post-event', seg_map_path='train/target'),
        pipeline=pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, img_suffix='.tif', seg_map_suffix='.tif',
        data_prefix=dict(img_path_from='val/pre-event', img_path_to='val/post-event', seg_map_path='val/target'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
