import os
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import json
from PIL import Image
import shutil
from shutil import copyfile

import sys
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
sys.path.append(Path(__file__))
import json

# print(Path(__file__).parents[3])

# 필요한 작업 -> annotations 속 segmentation값을 bbox값을 통해 변환하여 추가
def convert(ROOT, DATA, TRACKS, NEW_PATH):
    root_path = os.path.join(NEW_PATH, TRACKS[0])
    # train_path = os.path.join(NEW_PATH, TRACKS[0])
    # test_path = os.path.join(NEW_PATH, TRACKS[0])
    
    # print(train_path)
    
    tr_coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0,"name": "affiliation"},
                       {"id": 1,"name": "affiliation-marker"},
                       {"id": 2,"name": "author"},
                       {"id": 3,"name": "author-marker"},],
        "info": [],
    }
    
    te_coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0,"name": "affiliation"},
                       {"id": 1,"name": "affiliation-marker"},
                       {"id": 2,"name": "author"},
                       {"id": 3,"name": "author-marker"},],
        "info": [],
    }
    
    val_id_lst = []
    test_id_lst = []
    for i in range(0,1305): # 임의로 지정
        if i % 2 == 0: # valid일 경우
            # 이미지 복사
            ori_path = DATA['images'][i]['file_name']
            new_path = '/home/yejin/DiT_dir/unilm/dit/object_detection/DATASETS/convert_LS_final/valid'
            f_name = DATA['images'][i]['file_name'].split('/')[-1]
            destination_file = os.path.join(new_path, f_name)
            
            shutil.copy2(ori_path, destination_file)
            
            DATA['images'][i]['file_name'] = destination_file # file_name 변경
            
            tr_coco_data['images'].append(DATA['images'][i])
            val_id_lst.append(DATA['images'][i]['id'])
        
        else: # test_test일 경우
            # 이미지 복사
            ori_path = DATA['images'][i]['file_name']
            new_path = '/home/yejin/DiT_dir/unilm/dit/object_detection/DATASETS/convert_LS_final/test_test'
            f_name = DATA['images'][i]['file_name'].split('/')[-1]
            destination_file = os.path.join(new_path, f_name)
            
            shutil.copy2(ori_path, destination_file)
            
            DATA['images'][i]['file_name'] = destination_file # file_name 변경
            
            te_coco_data['images'].append(DATA['images'][i])
            test_id_lst.append(DATA['images'][i]['id'])
    
    tr_coco_data['info'] = DATA['info']
    te_coco_data['info'] = DATA['info']
    
    for num in range(len(DATA['annotations'])):
        if DATA['annotations'][num]['image_id'] in val_id_lst: # valid에 넣을 부분
            tr_coco_data["annotations"].append(DATA['annotations'][num])
        
        else: # test_test에 넣을 부분
            te_coco_data["annotations"].append(DATA['annotations'][num])
    
    with open(f"{root_path}/valid.json", "w") as f:
        json.dump(tr_coco_data, f)
        
    with open(f"{root_path}/test_test.json", "w") as f:
        json.dump(te_coco_data, f)

    
if __name__ == '__main__':
    file_path = f'{Path(__file__).parents[3]}/object_detection/DATASETS/convert_LS_final/test.json'
    target_dir = f'{Path(__file__).parents[3]}/object_detection/DATASETS/convert_LS_final/test'
    with open(file_path, 'r') as json_file: # JSON 파일 열기
        data = json.load(json_file)
        
    convert_data_path = f'{Path(__file__).parents[3]}/object_detection/DATASETS'
    TRACKS = ["convert_LS_final"] #["CONVERT"]
    SPLITS = ["valid", "test_test"] # ["train", "test"]
    for track in TRACKS:
        for split in SPLITS:
            if not os.path.exists(os.path.join(convert_data_path, track, split)): # 디렉토리 생성부분 추가
                os.makedirs(os.path.join(convert_data_path, track, split))
        
    convert(target_dir, data, TRACKS, convert_data_path)