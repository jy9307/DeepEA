#!/bin/bash
# ===== 변수 설정 =====
MODE=eval
DATASET=val_essay_qas_val_expert_0.json
CHECKPOINT=ver.260315_expert_2_ep10.pt

# ===== 실행 =====
python main.py --mode $MODE --target $DATASET --model_path $CHECKPOINT