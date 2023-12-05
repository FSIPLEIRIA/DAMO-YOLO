#!/usr/bin/env python3

import os

from damo.config import Config as MyConfig


class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.miscs.exp_name = os.path.split(
            os.path.realpath(__file__))[1].split('.')[0]
        self.miscs.eval_interval_epochs = 100
        self.miscs.ckpt_interval_epochs = 1
        # optimizer
        self.train.batch_size = 4
        self.train.base_lr_per_img = 0.001 / 64
        self.train.min_lr_ratio = 0.05
        self.train.no_aug_epochs = 2
        self.train.warmup_epochs = 1
        self.train.total_epochs = 15

        self.train.optimizer = {
            'name': "AdamW",
            'weight_decay': 1e-2,
            'lr': 4e-3,
            }

        # augment
        self.train.augment.transform.image_max_range = (416, 416)
        self.train.augment.transform.keep_ratio = False
        self.test.augment.transform.keep_ratio = False
        self.test.augment.transform.image_max_range = (416, 416)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 0.2
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)
        self.train.augment.mosaic_mixup.keep_ratio = False

        self.dataset.train_ann = ('fsoco_train_coco', )
        self.dataset.val_ann = ('fsoco_val_coco', )

        # backbone
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_nano_large.txt')
        TinyNAS = {
            'name': 'TinyNAS_mob',
            'net_structure_str': structure,
            'out_indices': (2, 4, 5),
            'with_spp': True,
            'use_focus': False,
            'act': 'silu',
            'reparam': False,
            'depthwise': True,
            'use_se': False,
        }

        self.model.backbone = TinyNAS

        GiraffeNeckV2 = {
            'name': 'GiraffeNeckV2',
            'depth': 0.5,
            'hidden_ratio': 0.5,
            'in_channels': [80, 160, 320],
            'out_channels': [80, 160, 320],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
            'depthwise': True,
        }

        self.model.neck = GiraffeNeckV2

        ZeroHead = {
            'name': 'ZeroHead',
            'num_classes': 5,
            'in_channels': [80, 160, 320],
            'stacked_convs': 0,
            'reg_max': 7,
            'act': 'silu',
            'nms_conf_thre': 0.03,
            'nms_iou_thre': 0.65,
            'legacy': False,
            'last_kernel_size': 1,
        }
        self.model.head = ZeroHead

        self.dataset.class_names = ['unknown_cone', 'yellow_cone', 'blue_cone', 'orange_cone', 'large_orange_cone']
