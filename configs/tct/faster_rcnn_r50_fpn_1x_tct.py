_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))

dataset_type = 'CocoDataset'

classes = ('normal', 'ascus', 'asch', 'lsil', 'hsil_scc_omn', 'agc_adenocarcinoma_em', 'vaginalis', 'monilia',
           'dysbacteriosis_herpes_act', 'ec')

# data_root = '/root/commonfile/TCTAnnotatedData/'
# data_root = '/root/userfolder/datasets/tct/'
# data_root = '/home/hezhujun/datasets/tct/'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/coco/TCT_JPEGImages/train30000-cat10.json',
        img_prefix=data_root),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/coco//TCT_JPEGImages/val10000-cat10.json',
        img_prefix=data_root),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/coco//TCT_JPEGImages/test10000-cat10.json',
        img_prefix=data_root))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
work_dir = "work_dir/tct/faster_rcnn_r50_fpn_1x_tct/"
