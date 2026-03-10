#!/bin/bash
# ===== 변수 설정 =====
MODE=eval
DATASET=mid_1_society_essay_qas_val.json
CHECKPOINT=essay_model251224.pt

# ===== 실행 =====
python main.py --mode $MODE --target $DATASET --model_path $CHECKPOINT