#!/bin/bash
# ===== Variable setup =====
MODE=pred_ensemble
DATASET="val_essay_qas_val_expert_0.json"
CHECKPOINT_A="essay_model_260316_expert_1_ep10.pt" # Update to the actual model A filename
CHECKPOINT_B="essay_model_260316_expert_2_ep1.pt" # Update to the actual model B filename
BATCH_SIZE=16

# ===== Execution =====
echo "Running Ensemble Prediction..."
echo "Model A: $CHECKPOINT_A"
echo "Model B: $CHECKPOINT_B"
echo "Target Dataset: $DATASET"
echo "-------------------------------------"

python main.py \
    --mode $MODE \
    --target $DATASET \
    --model_path $CHECKPOINT_A \
    --model_path2 $CHECKPOINT_B \
    --batch_size $BATCH_SIZE

echo "-------------------------------------"
echo "Prediction finished. Check the 'ensemble_result' directory for the JSON outputs."
