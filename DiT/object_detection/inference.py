import argparse

import cv2

from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from pathlib import Path
import glob
import os

# print(f"{Path(__file__).parents[4]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
m = torch.load(f"{Path(__file__).parents[4]}/models/dit/publaynet_dit-b_cascade.pth", map_location=device)
#print(m)

# def main():
#     parser = argparse.ArgumentParser(description="Detectron2 inference script")
#     parser.add_argument(
#         "--image_path",
#         help="Path to input image",
#         type=str,
#         required=True,
#     )
#     parser.add_argument(
#         "--output_file_name",
#         help="Name of the output visualization file.",
#         type=str,
#     )
#     parser.add_argument(
#         "--config-file",
#         default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )

#     args = parser.parse_args()

#     # Step 1: instantiate config
#     cfg = get_cfg()
#     add_vit_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     # cfg.merge_from_file(f"{Path(__file__).parent}/publaynet_configs/cascade/cascade_dit_base.yaml")
#     #print(Path(__file__).parent) # /home/yejin/DiT_dir/unilm/dit/object_detection
    
#     # Step 2: add model weights URL to config
#     # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/models/dit/publaynet_dit-b_cascade.pth"
#     # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[2]}/dit/0704_first_output/model_final.pth"
#     #"https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_mrcnn.pth"
#     cfg.merge_from_list(args.opts)
    
#     # Step 3: set device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg.MODEL.DEVICE = device

#     # Step 4: define model
#     predictor = DefaultPredictor(cfg)
    
#     # Step 5: run inference
#     # print(args.image_path)
#     img = cv2.imread(args.image_path)

#     md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
#     if cfg.DATASETS.TEST[0]=='icdar2019_test':
#         md.set(thing_classes=["table"])
#     else:
#         # md.set(thing_classes=["text","title","list","table","figure"])
#         md.set(thing_classes=["affiliation","affiliation-marker","author","author-marker"])

#     # print(img)
#     # print('### img.shape ### ',img.shape)
#     # print(img[:, :, ::-1].shape)
    
#     with torch.no_grad():
#         #output = model(input)
#         # print('### predictor(img) ###\n',predictor(img).keys()) # key값으로 'instances'만 존재
#         output = predictor(img)["instances"]
#     # output = predictor(img)["instances"]
#     # print('### output ###\n',type(output.to("cpu")))
    
#     v = Visualizer(img[:, :, ::-1],
#                     md,
#                     scale=1.0,
#                     instance_mode=ColorMode.SEGMENTATION)
#     result = v.draw_instance_predictions(output.to("cpu"))
#     # print('### result ###\n',result)
#     # print('### result ###\n',result.get_image())
#     result_image = result.get_image()[:, :, ::-1]

#     # step 6: save
#     cv2.imwrite(args.output_file_name, result_image)

# if __name__ == '__main__':
#     main()

def main(input_img_path, output_img_path):
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    # parser.add_argument(
    #     "--image_path",
    #     help="Path to input image",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "--output_file_name",
    #     help="Name of the output visualization file.",
    #     type=str,
    # )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_file(f"{Path(__file__).parent}/publaynet_configs/cascade/cascade_dit_base.yaml")
    #print(Path(__file__).parent) # /home/yejin/DiT_dir/unilm/dit/object_detection
    
    # Step 2: add model weights URL to config
    # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/models/dit/publaynet_dit-b_cascade.pth"
    # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/DiT_dir/unilm/dit/0705_result/first_OutPut/model_final.pth"
    # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/DiT_dir/unilm/dit/0705_result/second_OutPut/model_final.pth"
    # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[2]}/dit/0704_first_output/model_final.pth"
    # cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/DiT_dir/unilm/dit/0706_result/first_OutPut/model_final.pth"
    cfg.MODEL.WEIGHTS = f"{Path(__file__).parents[4]}/DiT_dir/unilm/dit/0706_result/second_OutPut/model_0003999.pth"
    #"https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_mrcnn.pth"
    # cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    
    # Step 5: run inference
    args.image_path = input_img_path
    img = cv2.imread(args.image_path)

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["affiliation","affiliation-marker","author","author-marker"])
        
    output = predictor(img)["instances"]
    
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    # print('### result ###\n',result)
    # print('### result ###\n',result.get_image())
    result_image = result.get_image()[:, :, ::-1]

    # step 6: save
    args.output_file_name = output_img_path
    cv2.imwrite(args.output_file_name, result_image)

if __name__ == '__main__':
    
    test_dataset = '/home/yejin/DiT_dir/unilm/dit/object_detection/DATASETS/convert_LS_final/valid'
    # print(os.listdir(f'{test_dataset}')[170:])
    paper_lst = os.listdir(f'{test_dataset}')
    paper_lst.sort()
    for idx in range(0, len(paper_lst), 6):
        out_img_name = paper_lst[idx]
        input_img_path = os.path.join(test_dataset, out_img_name) # os.path.join(test_dataset, input_img)
        #out_img_name = f"{input_img.split('.png')[0]}_infer.png" # f"{input_img_path.split('/')[-1]}_infer.png"
        out_img_name = f"{out_img_name.split('.png')[0]}_infer.png"
        # output_img_path = '/home/yejin/DiT_dir/unilm/dit/0705_result/infer/baseline_infer'
        # output_img_path = '/home/yejin/DiT_dir/unilm/dit/0706_result/infer/first_infer'
        output_img_path = '/home/yejin/DiT_dir/unilm/dit/0706_result/infer/second_infer'
        output_img_path = os.path.join(output_img_path, out_img_name)
        
        # print(f'{out_img_name} ### {output_img_path}\n')
        main(input_img_path, output_img_path)