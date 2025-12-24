# tools/logger.py

import torch


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