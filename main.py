import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import cohen_kappa_score
import numpy as np

from datetime import datetime
import json
import argparse 
import os
import time
from tqdm import tqdm

from tools.logger import TrainingLogger, log_batch

class DEADataset(Dataset) :
    def __init__(self, data_list, tokenizer, max_len=512, max_score=5) :
        """
        data_list -> JSON 객체 리스트
        tokenizer : BERT기반 허깅페이스 토크나이저
        """

        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_score = max_score

    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self, idx) :
        item = self.data[idx]

        student_answer = item['response']
        question = item['question']
        rubric_item = item['rubric']
        rubric_desc_5 = rubric_item['evaluation_5']
        rubric_name = rubric_item['name']
        rubric_text = f"문항 : {question} / 평가영역: {rubric_name} / 기준: {rubric_desc_5}"

        encoding = self.tokenizer(
            rubric_text,
            student_answer,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
            truncation=True,
            return_tensors="pt"
        )

        try :
            labels = torch.tensor(
                [
                    float(item['score'][0]) / self.max_score,
                    float(item['score'][1]) / self.max_score
                ]
            )
        
        except :
            labels = torch.tensor([])

        embeddings_ = {
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask' : encoding['attention_mask'].flatten(),
            'token_type_ids' : encoding['token_type_ids'].flatten(),
            'labels' : labels
        }

        return embeddings_
    

class RubricCrossEncoder(nn.Module) :
    def __init__(self, model_name, num_trait = 1) :
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.bert.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size //2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, num_trait),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids = None) :

        if token_type_ids is not None :
            outputs = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids
            )
        
        else :
            outputs = self.bert(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

        cls_output = outputs.last_hidden_state[:, 0 , :]

        scores = self.regressor(cls_output)

        return scores

def data_open(name, mode) :
    folder = "train_data" if mode == "train" else "val_data"
    target_path = os.path.join(folder, name)

    with open(target_path, "r", encoding='utf-8') as f :
        target = json.load(f)

    return target




def train_fn(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)

def eval_fn(model, dataloader, criterion, device):
    """
    모델의 성능을 검증하는 함수
    Weight update 없음 (torch.no_grad)
    """
    model.eval()
    total_loss = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            
            # Loss 계산
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.vstack(all_preds), np.vstack(all_labels)

def predict_fn(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            all_preds.append(outputs.cpu().numpy())
            
    return np.vstack(all_preds)


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    # mode에 eval 추가
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model_path", type=str, default="essay_model.pt")
    parser.add_argument("--target", type=str, required=True) # target 파일명 필수
    args = parser.parse_args()

    MODEL_NAME = "klue/roberta-base"
    MAX_LEN = 512
    BATCH_SIZE = 16
    NUM_TRAITS = 2
    MAX_SCORE = 5

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("▶ Device: MPS")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("▶ Device: CUDA")
    else:
        DEVICE = torch.device("cpu")
        print("▶ Device: CPU")

    # 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RubricCrossEncoder(MODEL_NAME, num_trait=NUM_TRAITS).to(DEVICE)

    # 데이터 로드
    target = data_open(args.target, args.mode)
    dataset = DEADataset(data_list = target, tokenizer = tokenizer, max_len=MAX_LEN)
    
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=(args.mode == "train"))

    # --- Mode별 실행 ---
    if args.mode == "train":
        print(f"--- Training Start ---")

        load_path = os.path.join("checkpoints", args.model_path)
        if os.path.exists(load_path):
            print(f"▶ 기존 체크포인트를 발견했습니다: {load_path}")
            model.load_state_dict(torch.load(load_path, map_location=DEVICE))
            print("▶ 가중치 로드 완료. 이어서 학습을 시작합니다.")
        else:
            print("▶ 기존 체크포인트가 없습니다. 처음부터 학습을 시작합니다.")
        
        # 1. 로거 초기화
        config_info = f"Batch: {BATCH_SIZE}, LR: {args.lr}, Epochs: {args.epochs}"
        logger = TrainingLogger(log_dir="logs", model_name=MODEL_NAME, config_str=config_info)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        
        # 전체 학습 시작 시간 측정
        total_start_time = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            avg_loss = train_fn(model, dataloader, optimizer, criterion, DEVICE)
        
            epoch_end = time.time()
            duration = epoch_end - epoch_start
        
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f}")
            logger.log_epoch(epoch + 1, args.epochs, avg_loss, duration)

            # ==========================================
            # [수정] 에포크마다 모델 저장 로직 추가
            # ==========================================
            # 파일명 예시: essay_model_251224_ep1.pt
            base_name = args.model_path.split(".")[0]
            date_str = datetime.now().strftime("%y%m%d")
            ckpt_name = f"{base_name}_{date_str}_ep{epoch+1}.pt"
            
            save_path = os.path.join("checkpoints", ckpt_name)
            
            torch.save(model.state_dict(), save_path)
            
            # 로그 및 출력
            logger.log(f"Checkpoint saved: {save_path}")
            print(f"   >> Checkpoint saved to {save_path}")
            # ==========================================

        total_duration = time.time() - total_start_time
        logger.finish(total_duration)


    elif args.mode == "eval":

        model_path = os.path.join("checkpoints", args.model_path)

        print(f"--- Evaluation Start ---")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Error: Model file '{model_path}' not found.")
            exit()

        criterion = nn.MSELoss()
        avg_loss, preds, labels = eval_fn(model, dataloader, criterion, DEVICE)
        
        print(f"▶ Evaluation Result")
        print(f"   MSE Loss : {avg_loss:.4f}")
        print(f"   RMSE     : {np.sqrt(avg_loss):.4f}")
        
        qwk_scores = []
        for i in range(NUM_TRAITS):

            p_int = np.rint(preds[:, i]).astype(int) 
            l_int = np.rint(labels[:, i]).astype(int)
            
            # weights='quadratic'이 핵심
            score = cohen_kappa_score(l_int, p_int, weights='quadratic')
            qwk_scores.append(score)
            print(f"   Trait {i+1} QWK : {score:.4f}")

        print(f"   Average QWK : {np.mean(qwk_scores):.4f}")
        # -------------------------
        
        print("\n[Sample Comparison (Pred vs Real)]")
        for i in range(min(3, len(preds))):
            print(f"   Sample {i+1}: {preds[i]} vs {labels[i]}")

    elif args.mode == "predict":
        print(f"--- Prediction Start ---")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print("Loaded pretrained model weights.")
        else:
            print("Warning: Predicting with initialized (random) weights.")

        predictions = predict_fn(model, dataloader, DEVICE)
        print("\nTop 5 Predictions:")
        print(predictions[:5])