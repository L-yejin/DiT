#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer

import sys
from pathlib import Path
print(Path(__file__).parent)
# wandb.init(project='dit_project')
# init config setting
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 5,
#   "batch_size": 128
# }

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    print(Path(__file__).parent)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    register publaynet first
    """
    register_coco_instances(
        "publaynet_train",
        {},
        "./publaynet_data/train.json",
        "./publaynet_data/train"
    )

    register_coco_instances(
        "publaynet_val",
        {},
        "./publaynet_data/val.json",
        "./publaynet_data/val"
    )

    register_coco_instances(
        "icdar2019_train",
        {},
        "data/train.json",
        "data/train"
    )

    register_coco_instances(
        "icdar2019_test",
        {},
        "data/test.json",
        "data/test"
    )

    # register_coco_instances(name, metadata, json_file, image_root)
    # register_coco_instances('LS_train', {}, f'{Path(__file__).parent}/LS_coco_ver/trackA_modern/train.json', f'{Path(__file__).parent}/LS_coco_ver/trackA_modern/train')
    # register_coco_instances('LS_test', {}, f'{Path(__file__).parent}/LS_coco_ver/trackA_modern/test.json', f'{Path(__file__).parent}/LS_coco_ver/trackA_modern/test')
    register_coco_instances('LS_train', {}, f'{Path(__file__).parent}/DATASETS/convert_LS_final/train.json', f'{Path(__file__).parent}/DATASETS/convert_LS_final/train')
    # register_coco_instances('LS_test', {}, f'{Path(__file__).parent}/DATASETS/convert_LS_final/test.json', f'{Path(__file__).parent}/DATASETS/convert_LS_final/test')
    register_coco_instances('LS_test', {}, f'{Path(__file__).parent}/DATASETS/convert_LS_final/valid.json', f'{Path(__file__).parent}/DATASETS/convert_LS_final/valid')
    
    
    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    import wandb
    with wandb.init(project='DiT_Project') as wandb_proj: # 0706_dit_project, 0705_dit_project
        trainer = MyTrainer(cfg)
        wandb.watch(trainer._trainer.model, log='all') # gradient
        trainer.resume_or_load(resume=args.resume)
        wandb_proj.config.update(args)
        
        #wandb.log({'Loss': loss_value, 'Accuracy': accuracy_value})
        return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    # args.config_file = f'{Path(__file__).parent}/publaynet_configs/cascade/cascade_dit_base.yaml'
    # args.opts = ['MODEL.WEIGHTS', f'{Path(__file__).parent.parent}/0705_result/second_OutPut/model_final.pth']
    # args.opts = ['MODEL.WEIGHTS', f'{Path(__file__).parent.parent}/publaynet_dit-b_cascade.pth']
    
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

    import wandb
    wandb.finish()