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

## 2.4 정규화와 피드 포워드 층
정규화: 딥러닝 모델에서 입력이 일정한 분포를 갖도록 만들어 학습이 안정적이고 빨라질 수 있도록 하는 기법
<br>
Fully connected layer인 feed forward층 이용하여 전체
### 2.4.1 층 정규화 이해하기
> 정규화란 모델이 각 입력 변수의 중요성을 적절히 반영하여 좀 더 정확한 예측을 할 수 있도록 모든 입력 변수가 비슷한 범위와 분포를 갖도록 조정하는 것.

>x를 norm_x로 정규화하는 식

    norm_x = (x-평균)/표준편차 #표준정규분포 공식과 동일
- Batch normalization: 미니 배치 사이에 정규화 수행, 주로 이미지 처리에 사용
    - 자연어 처리 시에는 sequence 길이가 제각각이므로 패딩 기법으로 길이 맞춰주는데, 이 상태로 batch normalization 수행하면 효과가 떨어짐.
- Layer normalization: 각 토큰 임베딩의 평균과 표준편차를 구해 정규화 수행
    - different layer normalization methods by operating sequence
        - post-norm: after each attention and feed-forward layer
        - pre-norm: before each --
    - nn.LayerNorm (by Pytorch)
### 2.4.2 피드 포워드 층
> 데이터의 특징을 학습하는 fully connected layer. 입력 텍스트 전체를 이해하는 역할 담당.

> 선형 층, 드롭아웃 층, 층 정규화, 활성함수로 구성됨.
> 예제 코드

    class PreLayerNormFeedForward(nn.Module):
        def __init__(self, d_model, dim_feedforward, dropout):
            super().__init__()
            self.linear1 = nn.Linear(d_model, dim_feedforward) # 선형 층 1
            self.linear2 = nn.Linear(dim_feedforward, d_model) # 선형 층 2
            self.dropout1 = nn.Dropout(dropout) # 드랍아웃 층 1
            self.dropout2 = nn.Dropout(dropout) # 드랍아웃 층 2
            self.activation = nn.GELU() # 활성 함수
            self.norm = nn.LayerNorm(d_model) # 층 정규화

        def forward(self, src):
            x = self.norm(src)
            x = x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
            x = self.dropout2(x)
            return x

## 2.5 인코더
인코더: multi-head attention, layer norm, feed forward 층이 반복되는 형태
> 예제 코드

        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward, dropout):
                super().__init__()
                self.attn = MultiheadAttention(d_model, d_model, nhead) # 멀티 헤드 어텐션 클래스
                self.norm1 = nn.LayerNorm(d_model) # 층 정규화
                self.dropout1 = nn.Dropout(dropout) # 드랍아웃
                self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout) # 피드포워드

            def forward(self, src):
                norm_x = self.norm1(src)
                attn_output = self.attn(norm_x, norm_x, norm_x)
                x = src + self.dropout1(attn_output) # 잔차 연결

                # 피드 포워드
                x = self.feed_forward(x)
                return x
이 과정 반복하면 됨.
## 2.6 디코더
> 마스크 멀티 헤드 어텐션 사용함
- 답지를 미리 보지 않기 위해 특정 시점 이전에 생성된 토큰까지만 확인할 수 있게 마스크 추가
> 크로스 어텐션: 인코더의 결과를 디코더가 활용하는 연산

## 2.7 BERT, GPT, T5 등 트랜스포머를 활용한 아키텍처

### 2.7.1 인코더를 활용한 BERT
> BERT:
- 인코더에 집중한 모델, 양방향 문맥을 활용해 텍스트 이해
### 2.7.2 디코더를 활용한 GPT
> GPT:
- 디코더에 집중한 모델, 텍스트 생성 작업, 단방향 방식
### 2.7.3 인코더와 디코더를 모두 사용하는 BART, T5
> 입력 테스트에 노이즈를 추가하고 노이즈가 제거된 결과를 생성하는 과제 수행

## 2.8 주요 사전 학습 메커니즘

### 2.8.1 인과적 언어 모델링
> 디코더 모델을 학습시키는 방법

### 2.8.2 마스크 언어 모델링
> 중간에 빠진 단어를 유추하는 방법

<br>

# 03. 트랜스포머 모델을 다루기 위한 허깅페이스 트랜스포머 라이브러리
## 3.1 허깅페이스 트랜스포머란
다양한 트랜스포머 모델을 통일된 인터페이스로 사용할 수 있도록 지원하는 오픈소스 라이브러리
<br>
transformer library와 dataset library가 있다.

## 3.2 허깅페이스 허브 탐색하기

### 3.2.1 모델 허브
### 3.2.2 데이터셋 허브
### 3.2.3 모델 데모를 공개하고 사용할 수 있는 스페이스
> 사용자가 자신의 모델 데모를 간편하게 공개할 수 있는 기능.
별도의 웹 개발 없이 웹 인터페이스로 공유 가능

> 리더보드: 다양한 오픈소스 LLM과 그 성능 정보 게시
## 3.3 허깅페이스 라이브러리 사용법 익히기

### 3.3.1 모델 활용하기
> 허깅페이스에서는 모델을 body와 head로 구분함
- 같은 body를 사용하면서 다른 작업에 사용할 수 있도록
- body, body+head, head

> class AutoModel: 모델의 바디를 불러옴

> class AutoModelForSequenceClassification: 헤드가 포함된 모델을 불러옴

### 3.3.2 토크나이저 활용하기

> 토크나이저: text to token, token to idx, adding special tokens in need

> class AutoTokenizer: 저장소의 토크나이저 불러옴
- tokenizer_config.json에 토크나이저 정보, 
- tokenizer.json에 vocab 정보

> 토크나이저 사용하기
- input_ids: list of token id
- attention_mask: 1-> genuine text, 0 -> padding
- token_type_ids: id of the sentence
- convert_ids_to_tokens: input_ids to tokens
- decode: input ids to sentence

### 3.3.3 데이터셋 활용하기
> load_dataset('name_of_dataset', 'name_of_subset')
- split='train': train 데이터만 가져옴

## 3.4 모델 학습시키기
실습 예제: 한국어 기사 제목 바탕으로 기사 카테고리 분류
### 3.4.1 데이터 준비
> 코드 예제

    #klue의 ynat데이터에서 train, validation 데이터 불러옴
    from datasets import load_dataset
    klue_tc_train = load_dataset('klue', 'ynat', split='train')
    klue_tc_eval = load_dataset('klue', 'ynat', split='validation')
    #필요 없는 칼럼 제거
    klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
    klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])
    #label_str 칼럼 추가
    def make_str_label(batch):
        batch['label_str'] = klue_tc_label.int2str(batch['label'])
        return batch
    klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)
### 3.4.2 트레이너 API를 사용해 학습하기
트레이너 API를 통해 데이터로더 준비, 로깅, 평가, 저장 등의 기능을 쉽게 수행할 수 있다.
### 3.4.3 트레이너 API를 사용하지 않고 학습하기
트레이너 API의 문제점: 내부 동작을 파악하기 어려움
> 과정1: 학습을 위한 모델과 토크나이저 준비
- 모델, 토크나이저 불러오고 tokenize_function 정의
- GPU로의 모델 이동을 수동으로 해야함

> 과정2: 데이터 전처리
- 데이터셋 토큰화 수행
- 칼럼 이름 변경+불필요 칼럼 제거
- DataLOader로 데이터셋을 배치 데이터로

> 과정3: 학습을 위한 함수 정의
- train(): 모델을 학습모드로 변경
- batch data 속의 input_ids, attention_mask, labels 를 각각 모델에 인자로 전달해 계산
- loss 구해서 backdrop 수행-> 모델 업데이트
> 과정4: 평가를 위한 함수 정의

> 과정5: 학습 수행
- for문을 통해 한 epoch 수행, AdamW옵티마이저

### 3.4.4 학습한 모델 업로드하기

## 3.5 모델 추론하기

### 3.5.1 파이프라인을 활용한 추론

작업 종류, 모델, 설정을 입력으로 받음. 예측 확률이 가장 높은 레이블과 확률 반환

### 3.5.2 직접 추론하기

> 과정
- 모델과 토크나이저 불러와 토큰화 수행
- 모델 추론 수행
- 가장 큰 예측 확률 갖는 클래스 추출, 결과로 변환

<br>

# 04. 말 잘 듣는 모델 만들기

GPT-3가 ChatGPT로 변하기까지 지시 데이터셋을 통한 학습 + 사용자의 선호 학습이 추가됨
## 4.1 코딩 테스트 통과하기: 사전 학습과 지도 미세 조정
### 4.1.1 코딩 개념 익히기: LLM의 사전 학습
> #Remind 언어 모델이란 다음에 올 단어의 확률을 예측하는 모델. 정답 토큰을 예측할 확률을 높이는 방향으로 학습 수행함.
### 4.1.2 연습문제 풀어보기: 지도 미세 조정
> 지도(supervised: 학습 데이터에 정답 포함) 미세 조정
- 요청의 형식을 적절히 해석
- 응답의 형태를 적절히 작성
- 요청과 응답이 잘 '정렬(allign)'되도록 추가 학습
- 지시 데이터셋: 사용자의 지시에 맞춰 응답한 데이터셋
> 문제: 지시 데이터셋의 양이 매우 적으므로 학습의 질이 낮다
- 해결1: 노가다 데이터 라벨링
- 해결2: 데이터 라벨링 시 데이터 구조화
    - 알파카 템플릿: general description, instruction, input, response 구조
> 기존 LLM과 동일하게 인과적 언어 모델링으로 학습하지만, 학습 데이터셋이 다르므로 기능도 달라짐.
### 4.1.3 좋은 지시 데이터셋이 갖춰야 할 조건
> 얼마나 많은 지시 데이터셋이 필요한가?
- Meta says: 1000개면 충분하다. 
    - 이렇게 학습된 LIMA가 알파카(52000개), GPT-4보다 성능 좋음
- 양보다 질
    - 지시사항의 다양성, 응답 데이터의 품질이 더 중요함
- Superficial alignment hypothesis: 모델의 지식이나 능력은 사전 학습 단계에서 완성되고, 
<br>
정렬 데이터를 통해서는 '정렬'만 추가로 학습함

> 지시 데이터의 품질
- 좋은 데이터를 선별해서 학습하면 해당 도메인의 성능을 높일 수 있음

## 4.2 채점 모델로 코드 가독성 높이기
### 4.2.1 선호 데이터셋을 사용한 채점 모델 만들기
> 가독성 비교 데이터셋
- 선호 데이터, 비선호 데이터 구분
    - 선호/비선호 여부는 상대적으로 결정
    - 선호 데이터에 비선호 데이터보다 높은 점수를 주도록 학습
    - 생성한 답변의 점수를 평가하는 리워드 모델
### 4.2.2 강화 학습: 높은 코드 가독성 점수를 향해
> 사람의 피드백을 활용한 강화 학습(RLHF)
- agent가 envirinment에서 action을 한다. 
<br>
action에 따라 state가 바뀌고 그에 대한 reward가 생기는데, 
<br>
agent는 reward를 최대화하는 방향으로 학습함. 
<br>
높은 점수를 받기 위한 일련의 행동을 episode라고 함
- 보상 해킹: 보상을 높게 받는 데에만 집중해 행동을 하지 않거나 간단한 행동만 하는 것

### 4.2.3 PPO: 보상 해킹 피하기
> 근접 정책 최적화(Proximal Preference Optimization)
- 참고 모델과 가까운 범위에서 리워드 모델의 높은 점수를 찾는 것.
- 참고 모델과의 거리가 일정 값 이상이면 배제
### 4.2.4 RLHF: 멋지지만 피할 수 있다면...
> 문제점: 
- 리워드모델의 성능이 좋지 않으면 성능이 떨어짐
- 참고 모델, 학습 모델, 리워드 모델이 필요해 리소스 낭비가 큼
## 4.3 강화 학습이 꼭 필요할까?
### 4.3.1 기각 샘플링: 단순히 가장 점수가 높은 데이터를 사용한다면?
> 지도 미세 조정을 마친 LLM을 통해 여러 응답을 생성하고, 그 중에서 리워드 모델이 가장 높은 점수를 준 응답을 모아 다시 지도 미세 조정 수행
- 라마-2에서, PPO 단계 이전에 기각 샘플링 도입함
### 4.3.2 DPO: 선호 데이터셋을 직접 학습하기
> RLHF vs DPO
- RLHF: 선호 데이터셋으로 리워드 모델 만들고, 언어 모델의 출력 평가하면서 강화 학습 진행
- DPO: 선호 데이터셋을 직접 언어 모델에 학습시킴
    - ex. 입력 프롬프트: "최고의 프로그래밍 언어는", 선호: '파이썬', 비선호: '자바' 
    <br>
    -> '파이썬' 예측 확률 증가, '자바' 예측 확률 감소하도록 학습
### 4.3.3 DPO를 사용해 학습한 모델들
> 더 효율적으로 선호 데이터셋을 구축할 수 있다면, 더 적은 비용으로 더 빠르게 사람의 선호를 반영한 LLM을 만들 수 있을 것이다.
- 제퍼-7B-베타: LLM이 생성한 결과를 AI로 평가해 선호-비선호 데이터 쌍 구축(distillated DPO)
- 뉴럴-챗-7B: GPT-3.5, GPT-4의 답변을 선호 데이터로, 라마-2-13B의 답변을 비선호 데이터로 사용
- 튈루-2: 700억개 파라미터 모델에서도 DPO 효과 검증

<br>

# 05. GPU 효율적인 학습
GPU: 단순한 곱셈을 동시에 여러 개 처리하는 데 특화된 처리 장치
<br>
메모리가 한정되어 있고, 가격이 비싸기 때문에 효율적으로 써야 함

## 5.1 GPU에 올라가는 데이터 살펴보기
### 5.1.1 딥러닝 모델의 데이터 타입
> 딥러닝 모델 자체가 학습과 추론 과정에서 GPU에 올라감.
- 딥러닝 모델 용량 = 파라미터 수 * 파라미터당 비트(바이트) 수
- 기존 fp32(32bit): 수의 세밀한 표현 but 용량이 너무 큼
- 현재 bf16 or fp16(16bit): 수의 세밀함은 떨어지나 용량 줄임
    - 일반적으로 bf16이 수 표현 더 잘함.
### 5.1.2 양자화로 모델 용량 줄이기
> 양자화 기술의 핵심은 더 적은 비트를 사용하면서도 원본 데이터의 정보를 최대한 손실 없이 유지하는 것
> 변환 방법
- 두 데이터 형식의 최대/ 최솟값 각각 대응
    - 데이터가 몰려있다면 양쪽 끝 데이터 낭비
- 존재하는 데이터의 최댓값 범위로 디응
    - 이상치가 있으면 취약함
- k개의 데이터를 묶은 블록 단위로 양자화 수행
- 퀀타일 방식: 입력 데이터 크기 순으로 등수 매겨 작은 데이터에 매칭하는 것
    - 매번 모든 입력 데이터 등수 확인하고 배치해야 하기 때문에 계산량, 메모리 낭비
### 5.1.3 GPU 메모리 분해하기
> In GPU, there are data such as
- model parameter
- gradient
- optimizer state
- forward activation: 역전파 수행하기 위해 저장하고 있는 값

## 5.2 단일 GPU 효율적으로 활용하기
### 5.2.1 그레이디언트 누적
> 적은 GPU 메모리로도 더 큰 배치 크기와 같은 효과를 얻을 수 있지만, 추가적인 순전파 및 역전파 연산을 수행해야 되므로 학습 시간이 증가됨
### 5.2.2 그레이디언트 체크포인팅
> 역전파를 진행하면서 사용이 끝난 데이터 삭제
> 순전파 과정에서 중간 데이터 삭제하고, 필요할 때 다시 계산
> 예네를 절충하는 방법이 그레디언트 체크포인팅
- 중간에 값들을 저장하는 것
- 추가적 순전파 계산으로 학습 시간 증가
## 5.3 분산 학습과 ZeRO
### 5.3.1 분산 학습
> GPU 여러개 사용하는 것
- 목적: 학습 속도 증진, 학습이 어려운 모델 다루기
- 데이터 병렬화: 여러 GPU에 각 모델 올리고, 학습 데이터 병렬로 처리
- 모델 병렬화: 모델을 여러 GPU에 나눠 올리는 방식
    - 파이프라인 병렬화: 층별로 나누기
    - 텐서 병렬화: 같은 층도 나누기 by 열 병렬화 혹은 행 병렬화
### 5.3.2 데이터 병렬화에서 중복 저장 줄이기(ZeRO)
> 동일한 모델을 여러 GPU에 올리는 것은 메모리 낭비
- 해결하기 위해 모델을 나눠 여러 GPU에 올리고, 할당된 연산만 하는 것.
## 5.4 효율적인 학습 방법(PEFT): LoRA
PEFT: 일부 파라미터만 학습하는 방법
### 5.4.1 모델 파라미터의 일부만 재구성해 학습하는 LoRA
> 모델 파라미터 재구성하는 방식. 
- 기존의 파라미터 고정, 새로운 저차원 행렬 구성
- 그래디언트 옵티마이저 상태 저장 메모리 감소
### 5.4.2 LoRA 설정 살펴보기
> 결정 사항 세 가지
- 새로운 차원 몇으로 할 지 정하기
- 추가한 파라미터를 기존의 파라미터에 얼마나 많이 반영할 지
- 어떤 파라미터를 재구성할 지
### 5.4.3 코드로 LoRA 학습 사용하기

## 5.5 효율적인 학습 방법(PEFT): QLoRA
LoRA에 양자화를 추가한 것
### 5.5.1 4비트 양자화와 2차 양자화
> 입력 데이터가 정규 분포를 따른다는 가정 하 면적을 나누고 할당 - 4비트 양자화 NF4
> 2차 양자화 (32비트 상수 효율적으로 저장하는 방법)
### 5.5.2 페이지 옵티마이저
> NVIDIA 통합 메모리를 통해 GPU가 CPU 메모리 공유하는 것
- 페이징: 가상 메모리에서, 운영체제가  램이 가득 차면 일부 데이터를 디스크로 옮기고, 필요할 때 다시 램으로 데이터 불러오는 것
- 페이지 옵티마이저는 페이징과 유사하게 작동

### 5.5.3 코드로 QLoRA 모델 활용하기

<br>

# 06. sLLM 학습하기
## 6.1 Text2SQL 데이터셋
### 6.1.1 대표적인 Text2SQL 데이터셋
> WikiSQL, Spider - 데이터베이서 정보+요청사항 
### 6.1.2 한국어 데이터셋
### 6.1.3 합성 데이터 활용
> 4개의 칼럼(db_id, context, question, answer)로 구성된 데이터셋
- db_id: 동일한 값을 갖는 테이블은 같은 도메인을 공유
- context: 테이블 정보
## 6.2 성능 평가 파이프라인 준비하기
GPT-4를 사용해 생성된 SQL이 정답인지 판단
### 6.2.1 Text2SQL 평가 방식
> Exact Match: 생성한 SQL이 문자열과 동일한지
- 문자열 동일하지 않으면 다르다고 판단
> Execution Accuracy: 쿼리 수행 가능한 DB 만들고, 쿼리 수행해 정답과 비교
- DB를 추가로 준비해야 함
> GPT 활용하는 방식
### 6.2.2 평가 데이터셋 구축
> 선별된 데이터(100개 내외) 사용
### 6.2.3 SQL 생성 프롬프트
> 지시사항과 데이터를 포함한 프롬프트, 미세 조정할 때도 동일하게 적용
- 학습시에는 SQL(정답) 포함된 상태로 사용, SQL 생성할 때는 비워놓고 사용
### 6.2.4 GPT-4 평가 프롬프트와 코드 준비
> 반복적으로 GPT_4에 API 요청을 보내야 함
- jsonl 파일을 이용해 OpenAI사의 오픈소스 이용하면 요청 제한 관리 및 비동기적 요청 송신 가능

## 6.3 실습: 미세 조정 수행하기
### 6.3.1 기초 모델 평가하기
> 과정
- LLM 추론에 사용할 프롬프트 생성
- 프롬프트를 입력해 SQL 생성하고 저장
- 평가에 사용할 jsonl 파일 만들고, GPT-4 API에 평가 요청 전달
### 6.3.2 미세 조정 수행
> 과정
- 학습 데이터 내려받고 데이터 전처리
- 학습에 사용할 프롬프트 생성 후 저장
- 지도 미세 조정 수행
- LoRA 어댑터와 기초 모델 병합
### 6.3.3 학습 데이터 정제와 미세 조정
> 애초에 원본이 GPT로 생성한 데이터이므로 잘못된 데이터가 섞여 있음.
- GPT-4를 이용해 SQL을 맥락, 요청 기준으로 확인하는 필터링 수행
### 6.3.4 기초 모델 변경
> 규모, 성능이 더 좋은 모델 사용하면 성능이 좋아짐,
### 6.3.5 모델 성능 비교
> sLLM임에도 불구하고 GPT-4와의 격차가 적음(특정 도메인 한정)











 