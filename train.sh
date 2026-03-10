#!/bin/bash
# ===== 변수 설정 =====
MODE=train
DATASET=mid_1_society_essay_qas_train.json


# ===== 실행 =====
python main.py --mode $MODE --target $DATASET --epochs 10 --model_path essay_model_251225_ep10_251225_ep10.pt