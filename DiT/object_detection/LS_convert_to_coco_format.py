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

# print(f'{Path(__file__).parents[1]}/object_detection/LS_coco_ver/result.json')
# file_path = f'{Path(__file__).parents[1]}/object_detection/LS_coco_ver/result.json'

# with open(file_path, 'r') as json_file: # JSON 파일 열기
#     data = json.load(json_file)
# print(len(data['annotations']))
# print(data['annotations'][0])
# print(data['annotations'][-1])

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
    
    te_num = 3
    te_num_lst=[]
    for i in range(0,6525): # 임의로 지정
        if i == te_num: # 여기 들어오는건 test
            te_num_lst.append(i)
            # print(DATA['images'][i])
            # 이미지 복사
            img_name = DATA['images'][i]['file_name'].split('/')[-1]
            
            source_file = os.path.join(ROOT, 'images', img_name)
            destination_file = f'{root_path}/test/{img_name}'
            # print(source_file)
            # print(destination_file)
            shutil.copy2(source_file, destination_file)

            DATA['images'][i]['file_name'] = destination_file # file_name 변경
            
            te_coco_data['images'].append(DATA['images'][i])
            te_num+=5

        else:
            # 이미지 복사
            img_name = DATA['images'][i]['file_name'].split('/')[-1]
            
            source_file = os.path.join(ROOT, 'images', img_name)
            destination_file = f'{root_path}/train/{img_name}'
            # print(source_file)
            # print(destination_file)
            shutil.copy2(source_file, destination_file)

            DATA['images'][i]['file_name'] = destination_file # file_name 변경
            
            tr_coco_data['images'].append(DATA['images'][i])
            
    #tr_coco_data['categories'] = DATA['categories']
    tr_coco_data['info'] = DATA['info']
    #te_coco_data['categories'] = DATA['categories']
    te_coco_data['info'] = DATA['info']
    
    # print(te_num_lst)
    for num in range(len(DATA['annotations'])):
        dic = {}
        if DATA['annotations'][num]['image_id'] in te_num_lst: # test에 넣을 부분
            x, y, w, h = DATA['annotations'][num]['bbox']
            dic['segmentation'] = [[x,y, x,y+h, x+w,y+h, x+w,y]]
        
            dic['area'] = DATA['annotations'][num]['area']
            dic['iscrowd'] = DATA['annotations'][num]['iscrowd']
            dic['image_id'] = DATA['annotations'][num]['image_id']
            dic['bbox'] = DATA['annotations'][num]['bbox']
            dic['category_id'] = DATA['annotations'][num]['category_id']
            dic['id'] = DATA['annotations'][num]['id'] 
            dic['ignore'] = DATA['annotations'][num]['ignore']
            
            te_coco_data["annotations"].append(dic)
        
        else: # train에 넣을 부분
            x, y, w, h = DATA['annotations'][num]['bbox']
            dic['segmentation'] = [[x,y, x,y+h, x+w,y+h, x+w,y]]
        
            dic['area'] = DATA['annotations'][num]['area']
            dic['iscrowd'] = DATA['annotations'][num]['iscrowd']
            dic['image_id'] = DATA['annotations'][num]['image_id']
            dic['bbox'] = DATA['annotations'][num]['bbox']
            dic['category_id'] = DATA['annotations'][num]['category_id']
            dic['id'] = DATA['annotations'][num]['id'] 
            dic['ignore'] = DATA['annotations'][num]['ignore']
            
            tr_coco_data["annotations"].append(dic)
    
    with open(f"{root_path}/train.json", "w") as f:
        json.dump(tr_coco_data, f)
        
    with open(f"{root_path}/test.json", "w") as f:
        json.dump(te_coco_data, f)
        




    
if __name__ == '__main__':
    # file_path = f'{Path(__file__).parents[1]}/object_detection/DATASETS/origin/LS_coco_ver2/result.json'
    # target_dir = f'{Path(__file__).parents[1]}/object_detection/DATASETS/origin/LS_coco_ver2'
    file_path = f'{Path(__file__).parents[1]}/object_detection/DATASETS/origin/LS_coco_ver_final/result.json'
    target_dir = f'{Path(__file__).parents[1]}/object_detection/DATASETS/origin/LS_coco_ver_final'
    with open(file_path, 'r') as json_file: # JSON 파일 열기
        data = json.load(json_file)
        
    convert_data_path = f'{Path(__file__).parents[1]}/object_detection/DATASETS'
    TRACKS = ["convert_LS_final"] #["CONVERT"]
    SPLITS = ["train", "test"]
    for track in TRACKS:
        for split in SPLITS:
            if not os.path.exists(os.path.join(convert_data_path, track, split)): # 디렉토리 생성부분 추가
                os.makedirs(os.path.join(convert_data_path, track, split))
        
    convert(target_dir, data, TRACKS, convert_data_path)