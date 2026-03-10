# 🖋️ DeepEssayAssessor (DeepEA)

**DeepEssayAssessor**는 BERT 기반의 **Cross-Encoder** 구조를 활용하여 서술형 답안을 자동 채점하는 딥러닝 모델입니다. '
문제(Question)와 채점 기준(Rubric)을 답안과 동시에 분석하여 정밀한 점수를 산출합니다.

---

## ✨ Key Features
* **Cross-Attention Mechanism**: 문제+루브릭과 학생 답안을 하나의 시퀀스로 입력하여 토큰 간의 깊은 상관관계를 학습합니다.
* **Sigmoid Scaling**: 점수를 0~1 사이로 정규화하여 학습 안정성을 높이고 출력 범위를 제어합니다.
* **QWK Metrics**: 채점 모델의 표준 평가지표인 Quadratic Weighted Kappa를 통해 신뢰도를 검증합니다.

---

- data_modifier.py는 


---
# History