# DiT
논문 정보 추출을 위한 DiT

---

origin DiT GitHub| https://github.com/microsoft/unilm/tree/master/dit

---

# How to Use  

## install Detectron2  
GitHub| https://github.com/facebookresearch/detectron2  

- 가상 환경 설치  
```conda create -n dit_py310 python=3.10```  
```conda activate dit_py310```  

- 본인의 cuda version에 맞춰 torch 설치 (현재 서버 cuda: 11.6)
```pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116```  

- detectron2 설치  
```python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'```  
```python -m pip install timm scipy opencv-python shapely```

## Git clone
```git clone https://github.com/cogcominc/article-ie.git```   
```cd article-ie```  

- dir_weight 폴더에 사용할 weight 추가
- dit/object_detection/DATASETS 폴더에 사용할 데이터 셋 추가
- dit/object_detection/train_net.py에 데이터 셋의 이미지와 json 파일의 경로를 사용하여 데이터 셋 추가 ex. line90,91  

#### article-ie 폴더 속 ```.env```파일 필요

## Training
```python dit/object_detection/train_net.py```  

- ```.env``` 속 아래 항목 추가 (아래 항목은 예시로 사용자에 맞춰 변경)
  1. config-file = 'object_detection/publaynet_configs/cascade/cascade_dit_base.yaml'
  2. num-gpus = 1
  3. MODEL.WEIGHTS = '/dir_weight/model_weight.pth'
  4. OUTPUT_DIR = '/RESULT/0717_result/first_OutPut'

## Inference
```python object_detection/inference.py```   

- ```.env``` 속 아래 항목 추가 (아래 항목은 예시로 사용자에 맞춰 변경. ```output_img_path```의 경우 우선 폴더 생성하기)
  1. config-file = 'object_detection/publaynet_configs/cascade/cascade_dit_base.yaml'
  2. MODEL.WEIGHTS = '/dir_weight/model_weight.pth'
  3. test_dataset = "/dit/object_detection/DATASETS/convert_LS_final/valid"
  4. output_img_path = "/RESULT/total_result/0716_first"

---

- training 시키는 경우

```python dit/object_detection/train_net.py --config-file dit/object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS <model weight 지정> OUTPUT_DIR <output 경로 지정>```  

MAX_ITER 등 config-file에 설정된 내용을 변경하고 싶을 경우 
    1. 해당 config-file로 이동하여 수정
    2. SOLVER.MAX_ITER 60000 등을 추가하여 사용해도 가능

- inference 하는 경우

```python object_detection/inference.py  --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml```  

1. inference.py 직접 수정하는 경우
    - inference.py 내에서 Step2에 해당하는 위치에 model weight 경로 입력
2. inference.py 실행 시 추가하는 경우
    - MODEL.WEIGHTS <model weight 지정> ← 해당 경로 지정해서 실행 시 같이 추가

3. 공통적으로 변경해야 하는 내용
    - line 146: validation set이 있는 위치 지정해줘야 함
    - line 158: inference 결과를 저장하는 위치 (실행 전 폴더 생성을 먼저 해야 함)
  
---



### cascade_dit_base 사용 & PubLayNet으로 fine-tuning된 weight를 초기 weight로 사용
##### 0705_second
- IMS_PER_BATCH: 4
- MAX_ITER: 3000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight: ./publaynet_dit-b_cascade.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./publaynet_dit-b_cascade.pth OUTPUT_DIR ./0705_result/second_OutPut``` 

##### 0707_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0705_result/second_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0705_result/second_OutPut/model_final.pth OUTPUT_DIR ./0707_result/first_OutPut```

##### 0708_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0707_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0707_result/first_OutPut/model_final.pth OUTPUT_DIR ./0708_result/first_OutPut```


##### 0708_second
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0708_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0708_result/first_OutPut/model_final.pth OUTPUT_DIR ./0708_result/second_OutPut```

##### 0709_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0708_result/second_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0708_result/second_OutPut/model_final.pth OUTPUT_DIR ./0709_result/first_OutPut```

##### 0711_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0709_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0711_result/first_OutPut```

---
### hyper-params 변경의 초기 weight는 0709_first의 weight 사용
#### optimizer 변경 test -> 성능 감소
- 'SGD'

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/third_OutPut```

- ‘ADAM’

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/fourth_OutPut```

- ‘RADAM’

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/fifth_OutPut```

---
#### Learning Rate 변경
- WARMUP_ITERS: 2000
- BASE_LR: 0.001

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0711_result/second_OutPut```

추가적으로 아래 내용으로 변경 후, 진행했지만 성능 감소
- WARMUP_ITERS: 1000, BASE_LR: 0.001
- WARMUP_ITERS: 500, BASE_LR: 0.001

---
### DiT의 base cascade의 weight 사용

##### 0713_second -> 총 56000 epoch 학습 됨
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./dit-base-224-p16-500k-62d53a.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ./0713_result/second_OutPut```

##### 0716_first
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0713_result/second_OutPut/model_0055999.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0713_result/second_OutPut/model_0055999.pth OUTPUT_DIR ./0716_result/first_OutPut```

---
##### DiT의 large cascade의 weight 사용 -> OOM 문제로 정지
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0001
- 초기 weight:  ./dit-large-224-p16-500k-d7a2fb.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_large.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_large.yaml --num-gpus 1 MODEL.WEIGHTS ./dit-large-224-p16-500k-d7a2fb.pth OUTPUT_DIR ./0715_result/first_OutPut```

