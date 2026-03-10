import os
import time
from datetime import datetime
import torch

# ==========================================
# 1. 학습 기록용 클래스 (File Logging)
# ==========================================

class TrainingLogger:
    def __init__(self, log_dir="logs", model_name="model", config_str=""):
        """
        학습 로그 파일을 생성하고 관리하는 클래스
        """
        # 로그 디렉토리 생성
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # 파일명 생성 (예: train_log_231225_1430.txt)
        self.file_name = f"train_log_{datetime.now().strftime('%y%m%d_%H%M')}.txt"
        self.file_path = os.path.join(log_dir, self.file_name)
        
        # 헤더 작성
        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write(f"=== Training Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Model: {model_name}\n")
            if config_str:
                f.write(f"Config: {config_str}\n")
            f.write("-" * 60 + "\n")
            
        print(f"▶ 로그 파일 생성됨: {self.file_path}")

    def log(self, message):
        """일반 텍스트를 파일에 기록"""
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_epoch(self, epoch, total_epochs, loss, duration_seconds):
        """에폭별 학습 결과를 포맷팅하여 기록"""
        mins, secs = divmod(duration_seconds, 60)
        time_str = f"{int(mins)}m {int(secs)}s"
        
        log_msg = f"Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Time: {time_str}"
        
        # 파일에 저장
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def finish(self, total_duration):
        """학습 종료 메시지 기록"""
        t_mins, t_secs = divmod(total_duration, 60)
        msg = f"\n=== Training Finished (Total: {int(t_mins)}m {int(t_secs)}s) ==="
        
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write("-" * 60 + "\n")
            f.write(msg + "\n")


def log_batch(
    batch: dict,
    tokenizer,
    batch_idx: int,
    max_tokens: int = 20,
    max_samples: int = 2,
    max_text_len: int = 500
):
    """
    DataLoader batch를 사람이 보기 좋게 로깅하는 함수

    Args:
        batch (dict): DataLoader에서 나온 batch
        tokenizer: HuggingFace tokenizer
        batch_idx (int): batch index
        max_tokens (int): attention_mask 등 앞부분 토큰 수
        max_samples (int): 디코딩해서 볼 샘플 개수
        max_text_len (int): 디코딩 텍스트 최대 길이
    """

    print("=" * 80)
    print(f"[Batch {batch_idx}]")

    # 1. Tensor summary
    print("\n[Tensor Summary]")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"- {k:15s} | shape={tuple(v.shape)} | dtype={v.dtype}")
        else:
            print(f"- {k:15s} | type={type(v)}")

    # 2. input_ids -> decoded text
    if "input_ids" in batch:
        print("\n[Decoded Text Samples]")
        input_ids = batch["input_ids"]
        for i in range(min(max_samples, input_ids.size(0))):
            text = tokenizer.decode(
                input_ids[i],
                skip_special_tokens=True
            )
            print(f"\n  Sample {i}")
            print("  " + text[:max_text_len])

    # 3. attention_mask head
    if "attention_mask" in batch:
        print("\n[Attention Mask Head]")
        print(batch["attention_mask"][:max_samples, :max_tokens])

    # 4. labels
    if "labels" in batch:
        print("\n[Labels]")
        print(batch["labels"][:max_samples])

    print("=" * 80)


def log_model_io(
    loss=None,
    logits=None,
    preds=None
):
    """
    모델 forward 결과 요약 로깅 (선택용)
    """
    print("\n[Model Output]")

    if loss is not None:
        print(f"- loss: {loss.item():.4f}")

    if logits is not None:
        print(f"- logits shape: {tuple(logits.shape)}")

    if preds is not None:
        print(f"- preds shape: {tuple(preds.shape)}")