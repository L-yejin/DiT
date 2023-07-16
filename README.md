# DiT
논문 정보 추출을 위한 DiT

---

# cascade_dit_base 사용 & PubLayNet으로 fine-tuning된 weight를 초기 weight로 사용
### 0705_second
- IMS_PER_BATCH: 4
- MAX_ITER: 3000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight: ./publaynet_dit-b_cascade.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./publaynet_dit-b_cascade.pth OUTPUT_DIR ./0705_result/second_OutPut``` 

### 0707_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0705_result/second_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0705_result/second_OutPut/model_final.pth OUTPUT_DIR ./0707_result/first_OutPut```

### 0708_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0707_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0707_result/first_OutPut/model_final.pth OUTPUT_DIR ./0708_result/first_OutPut```


### 0708_second
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0708_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0708_result/first_OutPut/model_final.pth OUTPUT_DIR ./0708_result/second_OutPut```

### 0709_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0708_result/second_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0708_result/second_OutPut/model_final.pth OUTPUT_DIR ./0709_result/first_OutPut```

### 0711_first
- IMS_PER_BATCH: 4
- MAX_ITER: 10000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0709_result/first_OutPut/model_final.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0711_result/first_OutPut```

---
# hyper-params 변경의 초기 weight는 0709_first의 weight 사
## optimizer 변경 test -> 성능 감소
- 'SGD'

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/third_OutPut```

- ‘ADAM’

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/fourth_OutPut```

- ‘RADAM’

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0712_result/fifth_OutPut```

---
## Learning Rate 변경
- WARMUP_ITERS: 2000
- BASE_LR: 0.001

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0709_result/first_OutPut/model_final.pth OUTPUT_DIR ./0711_result/second_OutPut```

추가적으로 아래 내용으로 변경 후, 진행했지만 성능 감소
- WARMUP_ITERS: 1000, BASE_LR: 0.001
- WARMUP_ITERS: 500, BASE_LR: 0.001

---
# DiT의 base cascade의 weight 사용

### 0713_second -> 총 56000 epoch 학습 됨
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./dit-base-224-p16-500k-62d53a.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ./0713_result/second_OutPut```

### 0716_first
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0004
- 초기 weight:  ./0713_result/second_OutPut/model_0055999.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_base.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./0713_result/second_OutPut/model_0055999.pth OUTPUT_DIR ./0716_result/first_OutPut```

---
# DiT의 large cascade의 weight 사용 -> OOM 문제로 정지
- IMS_PER_BATCH: 4
- MAX_ITER: 60000
- OPTIMIZER: ADAMW
- WARMUP_ITERS = 1000
- BASE_LR: 0.0001
- 초기 weight:  ./dit-large-224-p16-500k-d7a2fb.pth
- config-file: object_detection/publaynet_configs/cascade/cascade_dit_large.yaml

```python object_detection/train_net.py --config-file object_detection/publaynet_configs/cascade/cascade_dit_large.yaml --num-gpus 1 MODEL.WEIGHTS ./dit-large-224-p16-500k-d7a2fb.pth OUTPUT_DIR ./0715_result/first_OutPut```

