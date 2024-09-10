# Ch. 1 - LLM의 전반적인 내용

## ML과 DL의 차이:
ML은 특징 추출feature engineering 과정이 필요하지만 DL은 뉴럴넷이 스스로 특징들을 찾고 파라미터에 학습한다.

## 컴퓨터는 어떻게 단어의 의미를 이해하는가:
각 단어를 벡터화해서 다차원 공간에 저장한다. (수백~수만 차원을 사용함.) 이것을 word embedding이라 부름.

## Transfer Learning, Fine-Tuning
일반적인 데이터셋에 대해 사전학습pre-training을 하고 이후 특정 데이터셋에 대해 미세조정fine-tuning을 하면 특정 데이터셋에 대해서만 학습했을 때 보다 데이터의 특징을 더 잘 이해한다. 이 과정을 transfer learning이라고 한다. 

## Attention, Transformer
언어 모델의 문장 생성은 매 스텝마다 다음 단어(토큰)을 예측하며 이뤄짐. RNN을 사용했을 때는 순차적으로 직전의 단어에 가장 큰 영향을 받았지만 Attention mechanism을 사용하면 각 문장마다 (또는 해당 context window 내에서) 가장 '주의attention'을 줘야 하는 단어에 집중할 수 있다. 추가로, gradient vanishing/explosion 문제도 해결됨. Transformer architecture는 attention mechanism을 사용하는 모듈들을 사용한 구조를 말함.

## RLHF
다음 단어 예측은 잘 하지만 그렇다고 곧바로 좋은 챗봇이 되는 건 아님. (참고: System prompt를 받아서 이어서 완성한다는 점에서 챗봇의 역할을 하게 된다.) Supervised fine-tuning으로 인간이 직접 작성한 챗봇 스크립트를 학습시키고, Reward modeling으로 LLM이 작성한 답변을 인간이 직접 점수를 매겨 loss를 적용하고, 이 인간이 점수를 매기는 과정을 그대로 다른, 분리된 모델이 하도록 하는 과정을 말함.

## sLLM, 양자화, LoRA, RAG, 멀티모달
작은 모델. 파라미터의 숫자 표현을 fp32가 아니라 fp16, bp16, int8, int4 등으로 하는 것. 강한 양자화일수록 추론 단에서 하는 것이 성능을 지키는 방법. Low-Rank Adaptation. n, m이 큰 수일 때 n->m 차원의 변환을 n by m 행렬보다 n by a 행렬과 a by m 행렬로 계산하는 과정이 더 계산효율적임. Retrieval Augmented Generation. 벡터 데이터베이스에 불러오고자 하는 청크(문장/문단/문서)들을 임베딩 해 넣어놓고 관련 있는 청크들을 불러와서 생성 과정에 프롬프트로 넣음. 할루시네이션 억지에 효과적. LLM에 비전, 오디오도 함께.