# LLM을 활용한 실전 AI 애플리케이션 개발 | 1부 LLM의 기초 뼈대 세우기

# 01. LLM 지도

## 1.1 딥러닝과 언어 모델링

**LLM(Large Language Model)**: 딥러닝 기반 언어 모델(다음에 올 단어 예측하는 모델)

### 1.1.1 데이터의 특징을 스스로 추출하는 딥러닝

> 딥러닝에서 문제를 해결하는 방법

- 문제의 유형에 따라 일반적으로 사용되는 모델을 준비
- 풀고자 하는 문제에 대한 학습 데이터 준비
- 학습 데이터를 반복적으로 모델에 입력
> 딥러닝 vs 머신 러닝
- 머신 러닝: 데이터의 특징을 사람이 찾음
- 딥러닝: 데이터의 특징을 컴퓨터가 스스로 찾음 

Any ways to extract features from data by computer itself?
<br>

### 1.1.2 임베딩: 딥러닝 모델이 데이터를 표현하는 방식
> **임베딩**: 데이터의 의미와 특징을 포착해 숫자로 표현한 것

> 임베딩 작업 예시
- 검색 및 추천: 검색어와 관련 있는 상품 추천
- 클러스터링 및 분류: 유사하고 관련 있는 데이터 묶음
- 이상치 탐지: 나머지 데이터와 거리가 먼 데이터를 이상치로 판별

Then how to embed words?

> **word2vec**: 단어(word) 를 임베딩(즉, vector)로 변환하는 모델
- 일반적으로 n개의 숫자로 표현됨-> n dimensional vector
- 숫자의 의미를 파악하기 어려움
<br>

### 1.1.3 언어 모델링: 딥러닝 모델의 언어 학습법

 > **전이 학습(Transfer learning)**: 하나의 문제를 해결하는 과정에서 얻은 지식과 정보를 다른 문제를 풀 때 사용하는 방식

 - **사전 학습(pre-training)**: 대량의 데이터로 모델을 학습
 - **미세 조정(fine-tuning)**: 특정 문제 해결 위한 데이터로 추가 학습

 > 이미지 인식 분야에서 전이 학습 활용 예시
- **사전 학습 모델**: 동물 분류 모델
- **다운스트림(사전학습 모델 미세조정) 과제**: 유방암 판별
- **기존의 머신러닝 모델(지도학습 모델)과의 차이**
     - 기존 머신러닝: 각각의 데이터셋으로 별도 학습 진행
     - 전이 학습: 헤드 부분만 비교적 적은 데이터로 따로 학습
     - 데이터셋이 적어도 정확한 학습 가능
> 자연어 처리 분야에 적용(텍스트 분류 기능)
- 언어 모델 사전 학습
- 다운스트림 과제의 데이터셋으로 미세 조정
- 텍스트 분류 미세조정
- 시사점
    - 기존의 지도 학습보다 성능이 뛰어남
    - 텍스트 데이터 레이블 없이도 텍스트를 분류할 수 있음
> Transformer Architecture
- Biderectional Encoder Representations from Transformer
- Generative Pre-trained Transformer

<br>


## 1.2 언어 모델이 챗GPT가 되기까지
트랜스포머 아키텍쳐, GPT 그리고 정렬
### 1.2.1 RNN에서 트랜스포머 아키텍처로
> RNN의 특징: 하나의 hidden state에 지금까지의 입력 텍스트의 맥락을 압축
- 장점: 메모리 적게 사용, 다음 단어 빠르게 생성 가능
- 단점: 먼저 입력한 단어의 의미가 희석됨
> Transformer의 특징: 맥락/관계를 모두 계산. RNN의 문제점을 해결.
- 장점: 맥락 압축하지 않아 성능이 높음. 병렬 처리라 학습 속도 빠름 
- 단점: 메모리 사용량, 예측에 필요한 연산 시간 증가
<br>

### 1.2.2 GPT 시리즈로 보는 모델 크기와 성능의 관계
GPT 모델은 파라미터를 키워 가며 성능 개선함. 어떻게 가능했을까?
<br>
언어 모델의 학습 과정을 학습 데이터를 "손실 압축"하는 과정으로 해석 가능. 
<br>
원본 데이터의 크기가 커지면 성능 또한 증가(?)
### 1.2.3 챗GPT의 등장
> 기존 사람의 말을 이어 쓰는 기능만 있던 GPT-3에 사람의 요청사항에 답할 수 있는 기능을 탑재
- 정렬: LLM이 생성하는 답변을 사용자의 요청 의도에 맞추는 것
- 지도 미세 조정(Supervised fine-tuning): 지시 데이터셋(instruction dataset)으로 사전 학습 언어모델을 추가 학습
- Reinforcement Learning from Human Feedback: 선호 데이터셋으로 LLM을 평가하는 리워드 모델을 만들어 LLM이 더욱 높은 점수를 받을 수 있도록 추가 학습

<br>


## 1.3 LLM 애플리케이션의 시대가 열리다
LLM이 우리 생활에 미치는 영향: sLLM, 효율적인 학습과 추론, RAG를 중심으로
### 1.3.1 지식 사용법을 획기적으로 바꾼 LLM
> LLM은 다재다능하다. 
- 다재다능함이란 하나의 언어 모델이 다양한 작업에서 뛰어난 능력을 보이는 것
- 즉 하나의 LLM으로 여러 작업을 수행하는 것이 가능함
- 언어 이해와 언어 생성이 모두 가능함
> 지식 노동자가 수행하는 작업 대체할 능력이 있다.
### 1.3.2 sLLM: 더 작고 효율적인 모델 만들기
> LLM을 활용하는 방법
- **상업용 API**: 모델이 크고 범용 텍스트 생성 능력 뛰어남
- **오픈소스 LLM을 통해 직접 LLM API 생성해 사용**: 원하는 데이터로 자유롭게 추가 학습 가능
    - **sLLM**: 모델 크기 작으나 특정 도메인에서 높은 성능을 보이는 모델

### 1.3.3 sLLM: 더 작고 효율적인 모델 만들기
> 문제: LLM의 트랜스포머 아키텍쳐는 연산량이 엄청 남. 이것을 감당하기 위해 고가/고사양의 GPU가 필요함. 

> 해결책:
- 양자화: 모델 파라미터를 더 적은 비트로 표현
- **Low Rank Adaptation**: 모델의 일부만 학습
### 1.3.4 LLM의 환각 현상을 대처하는 RAG 기술
> 환각 현상(Hallucination) 발생 원인
- LLM은 어떤 정보가 참인지 거짓인지 학습한 적이 없음
- 학습 데이터 압축 과정에서 비교적 드물게 등장하는 정보의 소실
> 해결책:
- RAG(검색 증강 생성) 기술: 필요한 정보를 미리 추가하는 기술
## 1.4 LLM의 미래: 인식과 행동의 확장
## 1.5 정리

<br>


# 02. LLM의 중추, 트랜스포머 아키텍처 살펴보기
## 2.1 트랜스포머 아키텍처란
> 현재 트랜스포머는 자연어 처리를 포함한 모든 AI 분야의 핵심 아키텍처로 사용됨.

> 트랜스포머 아키텍처의 장점
- 확장성: 더 깊은 모델 만들어도 학습이 잘 됨.
- 효율성: 학습 시 병렬 연산이 가능하여 학습 시간 단축
- 더 긴 입력 처리: 어텐션 연산으로, 입력이 길어져도 성능이 거의 안 떨어짐
> 작동 방식
- 인코더
    - 토큰 임베딩 -> 위치 인코딩 -> 층 정규화 -> 멀티 헤드 어텐션 -> 피드 포워드 -> 디코더로 전달
- 디코더
    - 층 정규회 -> 멀티 헤드 어텐션 -> 크로스 어텐션 -> 피드 포워드

<br>

## 2.2 텍스트를 임베딩으로 변환하기
임베딩: 텍스트를 모델에 입력할 수 있는 숫자형 데이터
<br>
임베딩 과정: 토큰화 -> 토큰 임베딩 -> 위치 인코딩 추가
### 2.2.1 토큰화
> 토큰화: 텍스트를 적절한 단위로 나누고 숫자 아이디를 부여하는 것
- 작게는 자모 단위부터 크게는 단어 단위
- 어떤 토큰이 어떤 숫자 이이디로 연결되는지 사전(vocabulary)에 기록해 두어야 함.
    - 큰 단위 토큰화: 텍스트 의미 유지 용이 but 사전 수 커짐+OutOfVocab문제
    - 작은 단위 토큰화: 사전 크기 작아짐+OOV문제 해결 but 텍스트 의미 유지 불가
>서브 워드 토큰화: 데이터에 등장하는 빈도에 따라 토큰화 단위 결정
- 자주 나오는 단어는 단어 단위로 토큰화
    - 자주 사용하는 표현, 국가 이름, 유명인 이름
- 가끔 나오는 단어는 더 작은 단위로 토큰화
    - 외국어, 특수 문자, 이모티콘
> 예제 코드

    # 띄어쓰기 단위로 분리
    input_text = "나는 최근 파리 여행을 다녀왔다"
    input_text_list = input_text.split()
    # 토큰 -> 아이디 딕셔너리
    str2idx = {word:idx for idx, word in enumerate(input_text_list)}
    # 아이디 -> 토큰 딕셔너리
    idx2str = {idx:word for idx, word in enumerate(input_text_list)}
    # 토큰을 토큰 아이디로 변환
    input_ids = [str2idx[word] for word in input_text_list]
### 2.2.2 토큰 임베딩으로 변환하기
> 예제 코드


    import torch
    import torch.nn as nn
    embedding_dim = 16
    #nn.Embedding 클래스로 토큰 아이디를 토큰 임베딩으로 변환
    embed_layer = nn.Embedding(len(str2idx), embedding_dim)
    input_embeddings = embed_layer(torch.tensor(input_ids)) # (5, 16)
    input_embeddings = input_embeddings.unsqueeze(0) # (1, 5, 16)
아직 토큰의 의미를 담아 벡터로 전환하는 단계가 아니다. 그저 임의의 숫자로 바꿔준 것임.
<br>
딥러닝에서는 기존 머신러닝과 다르게 데이터의 의미를 담은 임베딩 생성 방법도 학습함.
### 2.2.3 위치 인코딩
> RNN과 달리 트랜스포머는 입력 데이터의 순서 정보를 추가해줘야 함. 그것이 위치 인코딩.
- 절대적 위치 인코딩: 입력 토큰 위치에 따라 고정된 임베딩 더해줌
- 상대적 위치 인코딩: 
> 절대적 위치 인코딩 예제 코드

    embedding_dim = 16
    max_position = 12
    # 위치 인코딩 층 생성
    position_embed_layer = nn.Embedding(max_position, embedding_dim) #16차원 위치 벡터 생성
    position_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)#0부터 input_ids.size(1)까지 1씩 증가하도록 데이터 생성
    position_encodings = position_embed_layer(position_ids)
    token_embeddings = embed_layer(torch.tensor(input_ids)) # (5, 16)
    token_embeddings = token_embeddings.unsqueeze(0) # (1, 5, 16)
    # 토큰 임베딩과 위치 인코딩을 더해 최종 입력 임베딩 생성
    input_embeddings = token_embeddings + position_encodings
## 2.3 어텐션 이해하기
어텐션: 입력한 텍스트에서 어떤 간어가 서로 관련되는지 '주의를 기울여' 파악
<br>
쿼리, 키, 값

### 2.3.1 사람이 글을 읽는 방법과 어텐션
> 어텐션은 사람이 단어 사이의 관계를 고민하는 과정을 딥러닝 모델이 수행할 수 있도록 모방한 연산.

> 과정:
- 단어와 단어 사이 관계 계산
    - 관련이 깊은 단어와 그렇지 않은 단어 구분
- 관련이 깊은 단어는 더 많이, 관련이 적은 단어는 더 적게 맥락 반영
### 2.3.2 쿼리, 키, 값 이해하기
> 용어 정리
- 쿼리: 입력하는 검색어
- 키: 쿼리와 관련이 있는지 계산하기 위해 문서가 가진 특징
    - 문서의 제목, 본문, 저자 이름 등등
- 값: 문서를 정렬하여 제공할 때의 문서  
> 단어의 맥락을 반영하는 방법
- 평균 방법: 단어를 모두 동등하게 반영
    - 관련이 깊은 단어 반영 불가
- 가까이 있는 단어에 높은 가중치를 두는 방법
    - 유연성이 떨어짐
- 관련도를 규칙이 아니라 데이터 자체에서 계산해야 함.
- 쿼리와 키 토큰을 토큰 임베딩으로 변환하여 계산하는 방법
    - 같은 단어끼리는 임베딩이 동일하여 관련도가 크게 계산됨 -> 주변 맥락 고려 어려움
    - 간접적 관련성은 반영이 어려움
- 토큰 임베딩을 변환하는 가중치 Wq, Wk 도입하는 방법
    - 학습 과정에서 정확도 높이는 방향으로 가중치 업데이트
### 2.3.3 코드로 보는 어텐션
> 쿼리, 키, 값 벡터를 만드는 예제 코드

    head_dim = 16
    # 쿼리, 키, 값을 계산하기 위한 변환
    # nn.Linear: 가중치 입히기
    weight_q = nn.Linear(embedding_dim, head_dim)
    weight_k = nn.Linear(embedding_dim, head_dim)
    weight_v = nn.Linear(embedding_dim, head_dim)
    # 변환 수행
    querys = weight_q(input_embeddings) # (1, 5, 16)
    keys = weight_k(input_embeddings) # (1, 5, 16)
    values = weight_v(input_embeddings) # (1, 5, 16)
> 스케일 점곱 방식의 어텐션 예제 코드

    
    from math import sqrt
    import torch.nn.functional as F
    def compute_attention(querys, keys, values, is_causal=False):
        dim_k = querys.size(-1) # 16
        scores = querys @ keys.transpose(-2, -1) / sqrt(dim_k) #쿼리와 키를 곱한다.
        #분산이 커지는 것을 방지하기 위해 임베딩 차원 수의 제곱근으로 나눈다
        weights = F.softmax(scores, dim=-1) #score합이 1이 되도록 softmax를 취해 weight로 바꾼다.
        return weights @ values #가중치와 값을 곱해 입력과 같은 형태의 출력 반환한다.
### 2.3.4 멀티 헤드 어텐션
여러 어텐션 연산을 동시에 적용하면 성능을 더 높일 수 있다.
> 멀티 헤드 어텐션 과정
- 쿼리, 키 값을 n_head로 split
- 각각의 어텐션 계산
- 입력과 같은 형태로 재변환
- 선형 층을 통과시키고 최종 결과 반환
> 예제 코드

    class MultiheadAttention(nn.Module):
        def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
            super().__init__()
            self.n_head = n_head
            self.is_causal = is_causal
            self.weight_q = nn.Linear(token_embed_dim, d_model)
            self.weight_k = nn.Linear(token_embed_dim, d_model)
            self.weight_v = nn.Linear(token_embed_dim, d_model)
            self.concat_linear = nn.Linear(d_model, d_model)

        def forward(self, querys, keys, values):
            B, T, C = querys.size()
            querys = self.weight_q(querys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            attention = compute_attention(querys, keys, values, self.is_causal)
            output = attention.transpose(1, 2).contiguous().view(B, T, C)
            output = self.concat_linear(output)
            return output

    n_head = 4
    mh_attention = MultiheadAttention(embedding_dim, embedding_dim, n_head)
    after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)
    after_attention_embeddings.shape
ㅇ
## 2.4 정규화와 피드 포워드 층
정규화: 딥러닝 모델에서 입력이 일정한 분포를 갖도록 만들어 학습이 안정적이고 빨라질 수 있도록 하는 기법
<br>
Fully connected layer인 feed forward층 이용하여 전체















 