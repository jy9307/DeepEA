#!/bin/bash
# ===== 변수 설정 =====
MODE=train
DATASET=train_essay_qas_train_expert_0.json


# ===== 실행 =====
python main.py --mode $MODE --target $DATASET --epochs 10 --batch_size 16