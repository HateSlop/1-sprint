# LLM을 활용한 실전 AI 애플리케이션 개발
------------------
# 1.LLM 지도

## 1.1 딥러닝과 언어 모델링
**LLM(Large Language Model)** - 딥러닝에 기반
사람의 언어를 컴퓨터가 이해하고 생성할 수 있도록 연구하는 **자연어 처리(natural language processing)**   
**언어 모델** - 다음에 올 단어가 무엇일지 예측하면서 문장을 하나씩 만들어 가는 방식으로 텍스트 생성

### 1.1.1 데이터의 특징을 스스로 추출하는 딥러닝   
문제를 해결하는 방법   
>문제의 유형에 따라 일반적으로 사용되는 모델을 준비   
>풀고자 하는 문제에 대한 학습 데이터 준비   
>학습 데이터를 반복적으로 모델에 입력   

딥러닝 / 머신러닝 - 데이터의 특징을 누가 뽑는가?   
머신러닝은 데이터의 특징을 연구자 또는 개발자가 추출 / 딥러닝은 모델이 스스로 데이터의 특징을 찾고 분류   

### 1.1.2 임베딩: 딥러닝 모델이 데이터를 표현하는 방식   
**임베딩** - 데이터의 의미와 특징을 포착해 숫자의 집합으로 표현하는 것   
거리를 계산할 수 있기 때문에 다음과 같은 작업에 적합 

    검색 및 추천: 검색어와 관련이 있는 상품을 추천
    클러스터링및 분류: 유사하고 관련이 있는 데이터를 하나로 분류
    이상치 탐지: 나머지 데이터와 거리가 먼 데이터는 이상치로 표현   

그림 1.6

### 1.1.3 언어 모델링: 딥러닝 모델의 언어 학습법   
**전이 학습** - 하나의 문제를 해결하는 과정에서 얻은 지식과 정보를 다룬 문제를 풀 때 사용하는 방식   
대량의 데이터로 모델을 학습시키는 **사전학습** / 특정한 문제를 해결하기 위한 데이터로 추가 학습하는 **미세 조정**   
특정한 데이터만으로 학습한 모델보다 사전 학습 모델의 일부를 가져와 활용했을 때 더 성능이 좋음   
그림 1.9   

## 1.2 언어 모델이 챗GPT가 되기까지
### 1.2.1 RNN에서 트랜스포머 아키택처로   
**RNN** - 입력하는 텍스트를 순차적으로 처리해서 다음 단어를 예측   
그림 1.12   
**트랜스포머 아키텍처**는 순차적인 처리 방식이 아닌, 맥락을 모두 참조하는 어탠션(attention)연산을 사용   
그림 1.14   
맥락을 압축하지 않고 그대로 활용하기 때문에 성능은 높아지는 대신 무겁고 비효율적인 연산을 사용   

### 1.2.2 GPT 시리즈로 보는 모델 크기와 성능의 관계
GPT-1 = 1억 1,700만개의 파라미터 > GPT-2 = 15억개 > GPT-3 = 1,750억개   
언어 모델이 학습하는 과정을 학습 데이터로 압축   
그림 1.18   
하지만 모델이 계속해서 커진다고 성능이 비례하진 않고 학습 데이터의 크기가 최대 모델 크기의 상한   

### 1.2.3 챗GPT의 등장   
GPT-3는 그저 사용자의 말을 이어서 작성하는 능력밖에 없음   
**지도 미세 조정**(supervised fine-tuning)과 **RLHF**(Reinforcement Learning from Human Feedback)을 통해 사용자의 요청을 해결할 수 있는 텍스트를 생성 가능   
정렬(alignment) - LLM이 생성하는 답변을 사용자의 요청 의도에 맞추는 것   
지도 미세 조정 - 언어 모델링으로 사전 학습한 언어 모델을 지시 데이터셋(instruction dataset)으로 추가 학습하는 것   
지시 데이터셋 - 사용자가 요청 또는 지시한 사항과 그에 대한 적절한 응답을 정리한 데이터셋   
OpenAI는 두 가지 답변 중 사용자가 더 선호하는 답변의 데이터셋을 선호 데이터셋(preference dataset)으로 정리하고 이를 답변으로 평가하는 리워드 모델(reward model)을 생성하여 더 높은 점수를 받을 수 있도록 추가 학습 진행   
위와 같은 강화 학습을 진행하는 것 - RLHF   

## 1.3 LLM 애플리케이션의 시대가 열리다
### 1.3.1 지식 사용법을 획기적으로 바꾼 LLM   
기존의 자연어 처리 접근 방식에는 언어 이해 모델과 언어 생성 모델을 각각 개별해 연결   
LLM은 하나로 연결되어 더 빠르고 다양한 작업에 활용 가능   
그림 1.22 23

### 1.3.2 sLLM: 더 작고 효율적으로 모델 만들기
LLM을 활용하는 방법
1. OpenAI의 GPT-4와 같이 상업용 API를 사용
2. 오픈소스 LLM을 활용해 직접 생성

**sLLM** - 추가 학습을 하는 경우 모델의 크기가 작아서 특정 도메인 데이터나 작업에서 높은 성능을 보이는 모델

### 1.3.3 더 효율적인 학습과 추론을 위한 기술
많은 연산량을 처리하기 위해서 GPU를 사용 - 상당 부분의 비용이 이에 발생   
더 적은 비트로 표현하는 **양자화**(quantization)과 모델의 일부만 학습하는 **LoRA**(Low Rank Adaptation)을 활용하여 연산을 개선   

### 1.3.4 LLM의 환각 현상을 대처하는 검색 증강 생성(RAG) 기술
**환각 현상** - LLM이 잘못된 정보나 실제로 존재하지 않는 정보를 만들어 내는 현상   
**검색 증강 생성**(Retrieval Augmented Generation) - 프롬프트에 필요한 정보를 미리 추가함으로써 잘못된 정보를 생성하는 문제 해결   

## 1.4 LLM의 미래: 인식과 행동의 확장
**멀티 모달**(multi modal) - 더 다양한 형식의 데이터를 입출력하도록 발전시킨 LLM   

## 1.5 정리
그림 1.27

# 2 LLM의 중추, 트랜스포머 아키텍처 살펴보기
## 2.1 트랜스포머 아키텍처란
기존의 RNN은 학습 속도가 느리고, 입력이 길어지면 먼저 입력한 토큰의 정보가 희석되면서 성능이 떨어짐   
트랜스포머는 셀프 어텐션(self-attention)이라는 개념을 도입   
셀프 어텐션 - 입력된 문장 내의 각 단어가 서로 어떤 관련이 있는지 계산해서 각 단어의 표현을 조정   
트랜스포머의 장점   
>확장성: 더 깊은 모델을 만들어도 학습이 잘됨. 동일한 블록을 반복해 사용하기 때문에 확장이 용이   
>효율성: 학습할 때 병렬 연산이 가능하기 때문에 학습 시간이 단축   
>더 긴 입력 처리: 입력이 길어져도 성능이 거의 떨어지지 않음   

 그림 2.2   
공통적으로 입력을 임베딩(embedding)층을 통해 숫자 집합인 임베딩으로 변환
위치 인코딩(positional encoding)층에서 문장의 위치 정보를 더함

## 2.2 텍스트를 임베딩으로 변환하기
### 2.2.1 토큰화
**토큰화**(tokenization) - 텍스트를 적절한 단위로 잘라 숫자형 아이디를 부여   
큰 단위를 기준으로 토큰화할수록 텍스트의 의미가 잘 유지되지만 사전의 크기가 커짐   
그림 2.4   
데이터의 등장하는 빈도에 따라 토큰화 단위를 결정하는 서브워드(subword) 토큰화 방식을 사용   
그림 2.5   

### 2.2.2 토큰 임베딩으로 변환하기
토큰이 의미를 담기 위해서는 최소 2개 이상의 숫자 집합인 벡터(vector)여야 함   
딥러닝에서는 모델이 특정 작업을 잘 수행하도록 학습하는 과정에서 데이터의 의미를 잘 담은 임베딩을 만드는 방법도 함께 학습   

### 2.2.3 위치 인코딩
트랜스포머는 모든 입력을 동시에 처리하기 때문에 텍스트에서 순서 정보를 추가하기 위해 위치 인코딩을 진행   
**절대적 위치 인코딩**(absolute position encoding) - 모델로 추론을 수행하는 시점에서는 입력 토큰의 위치에 따라 고정된 임베딩을 더함   
**상대적 위치 인코딩**(relative position encoding) - 긴 텍스트를 추론하는 경우 절대적 위치 인코딩의 성능이 떨어져 상대적인 위치 정보를 더함   
그림 2.7   

## 2.3 어텐션 이해하기
### 2.3.1 사람이 글을 읽는 방법과 어텐션
**어텐션** - 사람이 단어 사이의 관계를 고민하는 과정을 딥러닝 모델이 수행할 수 있도록 모방한 연산   
단어와 단어 사이의 관계를 계산해서 그 값에 따라 관련이 깊은 단어와 그렇지 않은 단어를 구분   

### 2.3.2 쿼리, 키, 값 이해하기
위와 같은 과정을 처리하기 위해 정보 검색의 개념에서 쿼리, 키, 값이라는 개념을 도입   
>쿼리: 우리가 입력하는 검색어   
>키: 쿼리와 관련이 있는지 계산하기 위해 문서가 가진 특징   
>값: 쿼리와 관련이 깊은 키를 가진 문서를 찾아 관련도순으로 정렬한 문서   

그림 2.9

원하는 결과를 얻기 위해 쿼리와 키 토큰을 토큰 임베딩을 변환하여 계산   
- 같은 단어끼리는 무조건 임베딩이 동일하게 발생한다는 문제   
- 간접적인 관련성은 반영되기 어려움

토큰 임베딩을 변환하는 가중치를 도입   
쿼리, 키, 값 세 가지 가중치를 통해 내부적으로 토큰과 토큰 사이의 관계를 계산해서 적절히 주변 맥락을 반영   
그림 2.14   

### 2.3.3 코드로 보는 어텐션
```
class AttentionHead(nn.Module):
  def __init__(self, token_embed_dim, head_dim, is_causal=False):
    super().__init__()
    self.is_causal = is_causal
    self.weight_q = nn.Linear(token_embed_dim, head_dim) # 쿼리 벡터 생성을 위한 선형 층
    self.weight_k = nn.Linear(token_embed_dim, head_dim) # 키 벡터 생성을 위한 선형 층
    self.weight_v = nn.Linear(token_embed_dim, head_dim) # 값 벡터 생성을 위한 선형 층

  def forward(self, querys, keys, values):
    outputs = compute_attention(
        self.weight_q(querys),  # 쿼리 벡터
        self.weight_k(keys),    # 키 벡터
        self.weight_v(values),  # 값 벡터
        is_causal=self.is_causal
    )
    return outputs

attention_head = AttentionHead(embedding_dim, embedding_dim)
after_attention_embeddings = attention_head(input_embeddings, input_embeddings, input_embeddings)
```

### 2.3.4 멀티 헤드 어텐션
멀티 헤드 어텐션 - 한번에 여러 어텐션 연산을 동시에 적용하여 성능을 더 향샹   
그림 2.17   

## 2.4 정규화와 피드 포워드 층
**정규화** - 딥러닝 모델에서 입력이 일정한 분포를 갖도록 만들어 학습이 안정적이고 빨라질 수 있도록 하는 기법   
과거에는 배치 입력 데이터 사이에 정규화를 수행하는 **배치 정규화**(batch 
트랜스포머는 특징 차원에서 정규화를 수행하는 **층 정규화**(layer normalization)사용   
전체 입력 문장을 이해하는 연산을 위해 **완전 연결 층**(fully connected layer)인 피드 포워드 층을 사용   

### 2.4.1 층 정규화 이해하기
**층 정규화** - 데이터 분포가 서로 다르면 정확한 예측을 어렵게 만드는데, 이 데이터를 정규화하여 모든 입력 변수가 비슷한 범위와 분포를 갖도록 하는 것   
이미지 처리에서는 배치 정규화를 사용하고 자연어 처리에서는 층 정규화를 사용   
그림 2.18
그림 2.19   
**사전 정규화**(pre-norm) - 층 정규화를 적용하고 어텐션과 피드 포워드 층을 통과했을 때 학습이 더 안정적   

### 2.4.2 피드 포워드 층
피드 포워드 층(feed forward layer) - 데이터의 특징을 학습하는 완전 연결 층(fully connected layer)을 말함   
선형 층, 드롭아웃 층, 층 정규화, 활성 함수로 구성됨   

## 2.5 인코더
멀티 헤드 어텐션, 층 정규화, 피드 포워드 층이 반복되는 형태   
```
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
```

## 2.6 디코더
디코더 블록에서는 **마스크 멀티 헤드 어텐션**과 **크로스 어텐션**(cross attention) 사용   
앞에서 생성한 토큰을 기반으로 다음 토큰을 생성 - 인과적(causal) 또는 자기 회귀적(auto-regressive)   
학습할 때는 인코더와 디코더 모두 완성된 텍스트를 입력받기 때문에 특정 시점에서는 그 이전에 생성된 토큰까지만 확인할 수 있도록 마스크를 추가   
## 2.7 BERT, GPT, T5 등 트랜스포머를 활용한 아키텍처
표 2.1

### 2.7.1 인코더를 활용한 BERT
BERT(Bidirectional Encoder Representations from Transformers) - 구글에서 개발한 트랜스포머의 인코더만을 활용해 자연어 이해 태스크에 집중한 대표적인 모델   
- 입력 토큰의 일부를 마스크 토큰으로 대체하고 그 마스크 토큰을 맞추는 마스크 언어 모델링을 통해 사전 학습
- 양방향 문맥을 이해할 수 있다는 특징이 있어 자연어 이해 작업에서 뛰어난 성능

### 2.7.2 디코더를 활용한 GPT
GPT(Generative Pre-trained Transformer) - OpenAI에서 개발한 생성 작업을 위해 만든 모델
- 디코더만을 사용하여 생성 작업의 경우 입력 토큰이나 이전까지 생성한 토큰만을 문맥으로 활용하는 인과적 언어 모델링(Casual Language Modeling)을 사용하기 때문에 단방향 방식임

### 2.7.3 인코더와 디코더를 모두 사용하는 BART, T5
BART - 메타가 개발한 BERT와 GPT의 장점을 결합한 모델
- BERT보다 다양한 사전 학습 과제를 도입했고 더 자유로우누 변형 추가가 가능하다는 점에 차이가 있음   
T5 - 구글이 개발하였으며 모든 자연어 처리 작업이 결국 '텍스트에서 텍스트(Text to Text)로의 변환'이라는 아이디어를 바탕으로 함

## 2.8 주요 사전 학습 메커니즘
## 2.8.1 인과적 언어 모델링
인과적 언어 모델링 - 문장의 시작부터 끝까지 순차적으로 단어를 예측하는 방식   

## 2.8.2 마스크 언어 모델링
마스크 언어 모델링 - 입력 단어의 일부를 마스크 처리하고 그 단어를 맞추는 작업으로 모델을 학습   

# 3 트랜스포머 모델을 다루기 위한 허깅페이스 트랜스포머 라이브러리
## 3.1 허깅페이스 트랜스포머란
허깅페이스(Huggingface)팀이 개발한 트랜스포머(Transformers) 라이브러리는 공통된 인터페이스로 트랜스포머 모델을 활용할 수 있도록 지원함으로써 현재는 딥러닝 분야의 핵심 라이브러리가 됨

## 3.2 허깅페이스 허브 탐색하기
### 3.2.1 모델 허브
모델 허브에는 어떤 작업에 사용하는지, 어떤 언어로 학습된 모델인지 등 다양한 기준으로 모델이 분류되어 있음   

### 3.2.2 데이터셋 허브
모델 허브와 비슷하지만 데이터셋 크기, 데이터 유형 등이 추가로 있음   
**한국어 언어 이해 평가**(Korean Language Understanding Evaluation) - 대표적인 한국어 데이터셋 중 하나로 텍스트 분류, 기계 독해, 문장 유사도 판단 등 다양한 작업에서 모델의 성능을 평가하기 위해 개발된 벤치마크 데이터셋

### 3.2.3 모델 데모를 공개하고 사용할 수 있는 스페이스
스페이스는 사용자가 자신의 모델 데모를 간편하게 공개할 수 있는 기능   
다양한 오픈소스 LLM과 그 성능 정보를 게시하는 리더보드가 존재   

## 3.3 허깅페이스 라이브러리 사용법 익히기
### 3.3.1 모델 활용하기
허깅페이스에서는 모델을 **바디**(body)와 **헤드**(head)로 구분   
같은 바디를 사용하면서 다른 작업에 사용할 수 있도록 만들기 위함   
라이브러리에서는 제한없이 바디만, 헤드와 함께, 헤드가 함께 있는 모델의 바디만 불러올 수도 있음   

### 3.3.2 토크나이저 활용하기
**토그나이저** - 텍스트를 토큰 단위로 나누고 각 토큰을 대응하는 토큰 아이디로 변환   
```
tokenized = tokenizer("토크나이저는 텍스트를 토큰 단위로 나눈다")
print(tokenized)
# {'input_ids': [0, 9157, 7461, 2190, 2259, 8509, 2138, 1793, 2855, 5385, 2200, 20950, 2],
#  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))
# ['[CLS]', '토크', '##나이', '##저', '##는', '텍스트', '##를', '토', '##큰', '단위', '##로', '나눈다', '[SEP]']

print(tokenizer.decode(tokenized['input_ids']))
# [CLS] 토크나이저는 텍스트를 토큰 단위로 나눈다 [SEP]

print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))
# 토크나이저는 텍스트를 토큰 단위로 나눈다
``` 
토근화 결과 중 **token_type_ids**는 문장을 구분하는 역할   
**attention_mask**는 해당 토큰이 패딩 토큰인지 실제 데이터인지에 대한 정보

### 3.3.3 데이터셋 활용하기
```
from datasets import load_dataset
# 로컬의 데이터 파일을 활용
dataset = load_dataset("csv", data_files="my_file.csv")

# 파이썬 딕셔너리 활용
from datasets import Dataset
my_dict = {"a": [1, 2, 3]}
dataset = Dataset.from_dict(my_dict)

# 판다스 데이터프레임 활용
from datasets import Dataset
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3]})
dataset = Dataset.from_pandas(df)
```

## 3.4 모델 학습시키기
### 3.4.1 데이터 준비
```
train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']
```

### 3.4.2 트레이너 API를 사용해 학습하기
허깅페이스는 학습에 필요한 다양한 기능을 학습 인자(Training Arguments)만으로 쉽게 활용할 수 있는 트레이너 API를 제공   
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(test_dataset) # 정확도 0.84
```   
데이터셋을 준비하고 학습 인자를 설정하는데 필요한 몇 줄의 코드만으로도 모델 학습 가능   

### 3.4.3 트레이너 API를 사용하지 않고 학습하기
```
num_epochs = 1
optimizer = AdamW(model.parameters(), lr=5e-5)

# 학습 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer)
    print(f"Training loss: {train_loss}")
    valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
    print(f"Validation loss: {valid_loss}")
    print(f"Validation accuracy: {valid_accuracy}")

# Testing
_, test_accuracy = evaluate(model, test_dataloader)
print(f"Test accuracy: {test_accuracy}") # 정확도 0.82
```   
Trainer를 사용하면 간편하다는 장점이 있고, 사용하지 않으면 내부 동작을 명확히 할 수 있고 직접 학습 과정을 조절할 수 있음   

### 3.4.4 학습한 모델 업로드하기   
```
from huggingface_hub import login

login(token="본인의 허깅페이스 토큰 입력")
repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
# Trainer를 사용한 경우
trainer.push_to_hub(repo_id)
# 직접 학습한 경우
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
```

## 3.5 모델 추론하기
### 3.5.1 파이프라인을 활용한 추론
허깅페이스는 토크나이저와 모델을 결합해 데이터의 전후처리와 모델 추론을 간단하게 수행하는 pipeline을 제공   
```
from transformers import pipeline

model_id = "본인의 아이디 입력/roberta-base-klue-ynat-classification"

model_pipeline = pipeline("text-classification", model=model_id)

model_pipeline(dataset["title"][:5])
```

### 3.5.2 직접 추론하기
```
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomPipeline:
    def __init__(self, model_id):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def __call__(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits

        probabilities = softmax(logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

        return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

custom_pipeline = CustomPipeline(model_id)
custom_pipeline(dataset['title'][:5])
```

# 4 말 잘 듣는 모델 만들기
OpenAI는 요청과 답변 형식으로 된 지시 데이터셋을 통해 GPT-3가 사용자의 요청에 응답할 수 있도록 학습하고 사용자가 더 좋아하고 도움이 되는 답변을 생성할 수 있도록 추가 선호 학습을 진행   

## 4.1 코딩 테스트 통과하기: 사전 학습과 지도 미세 조정
### 4.1.1 코딩 개념 익히기: LLM의 사전 학습
다음 단어를 예측하는 언어 모델을 학습시킬 때는 학습 데이터의 일부를 입력으로 넣고 바로 다음에 나오는 정답 토큰을 맞추도록 학습   
그림 4.3   

### 4.1.2 연습문제 풀어보기: 지도 미세 조정
**지도 미세 조정**(supervised fine-tuning) - 요청의 형식을 적절히 해석하고, 응답의 형태를 적절히 작성하며, 요청과 응답이 잘 연결되도록 추가 학습   
**지시 데이터셋**(instruction dataset) - 사용자의 지시에 맞춰 응답한 데이터셋   
2023년 스탠퍼드대학교에서 오픈 소스 라마(Llama) 모델을 추가 학습한 알파카(Alpaca) 데이터셋   
- **지시사항**(instruction) / **입력**(input) / **출력**(output) / 앞의 것들을 정해진 포맷 하나로 묶은 데이터인 **텍스트**(text)

```
{
    "instruction": "Create a classification task by clustering the given list of items.",
    "input": "Apples, oranges, bananas, strawberries, pineapples",
    "output": "Class 1: Apples, Oranges\nClass 2: Bananas, Strawberries\nClass 3: Pineapples"
}
```

### 4.1.3 좋은 지시 데이터셋이 갖춰야 할 조건
메타에선 라마 모델을 정렬하는데 선별한 1,000개 정도의 지시 데이터셋 리마(LIMA)로 사전 학습이 가능하다고 발표   
- "지시 데이터셋에서 지시사항이 다양한 형태로 되어 있고 응답 데이터의 품질이 높을수록 정렬한 모델의 답변 품질이 높아진다."

'**피상적인 정렬 가설**(superficial alignment hypothesis)' 주장 - 모델의 지식이나 능력은 사전 학습 단계에서 대부분 학습하고 정렬 데이터를 통해서는 답변의 형식이나 모델이 능력과 지식을 어떻게 나열할지 정도만 추가로 배우기 때문에 적은 정렬 데이터로도 사용자가 원하는 형태의 답변을 생성할 수 있다는 가설 

메타와 마이크로소프트의 연구를 통해 좋은 지시 데이터셋이 갖춰야하는 조건을 정리

> 지시 데이터셋을 작은 규모로 구축하더라도 모델이 지시사항의 형식을 인식하고 답변할 수 있도록 만들 수 있다     
> 지시사항이 다양한 형태이고 답변의 품질이 높을수록 모델의 답변품질도 높아진다      
> 학습 데이터의 품질을 높이기 위해 모델의 목적에 맞춰 학습 데이터의 교육적 가치를 판단하고 교육적 가치가 낮은 데이터를 필터링하는 방법을 사용할 수 있다      
> 교재의 예제 데이터와 같은 고품질의 데이터를 학습 데이터에 추가하면 성능을 크게 높일 수 있다      

## 4.2 채점 모델로 코드 가독성 높이기
### 4.2.1 선호 데이터셋을 사용한 채점 모델 만들기
같은 형태의 두 데이터 중 사람이 더 선호하는 데이터를 선호 데이터(chosen data) 그렇지 않은 코드를 비선호 데이터(rejected data)라고 함   
데이터의 정보를 객관적인 점수로 변환하기 어렵기 때문에 선호 데이터에 비선호 데이터보다 높은 점수를 주도록 채점 모델을 학습   

### 4.2.2 강화 학습: 높은 코드 가독성 점수를 향해
강화 학습에서는 **에이전트**(agent)가 **환경**(environment)에서 **행동**(action)을 하는데 그 행동에 따라 환경의 **상태**(state)가 바뀌고 이에 따른 **보상**(reward)가 주어짐   
이때 에이전트는 가능하면 더 많은 보상을 받기 위해서 행동하고 이를 연속적으로 수행하는 행동의 모음을 **에피소드**(episode)라고 함   
그림 4.11   
이때 보상을 높게 받는 데에만 집중하는 보상 해킹(reward hacking)이 발생할 수 있음   

### 4.2.3 PPO: 보상 해킹 피하기
**보상 해킹** - 평가 모델의 높은 점수를 받는 과정에서 다른 능력이 감소하거나 평가 점수만 높게 받을 수 있는 우회로를 찾는 현상   
**근접 정책 최적화**(Proximal Preference Optimization) - 지도 미세 조정 모델을 기준으로 학습하는 모델이 너무 멀지 않게 가까운 범위에서 리워드 모델의 높은 점수를 찾도록 함   
그림 4.15   

### 4.2.4 RLHF: 멋지지만 피할 수 있다면...
RLHF를 사용하기 위해선 리워드 모델을 학습시켜야 하는데, 리워드 모델의 성능이 좋지 않으면 LLM이 일관성 없는 점수를 학습하게 됨   
따라서 성능이 높고 일관성 있는(robust) 리워드 모델을 만들어야 함   

## 4.3 강화 학습이 꼭 필요할까?
### 4.3.1 기각 샘플링: 단순히 가장 점수가 높은 데이터를 사용한다면?
**기각 샘플링** - 지도 미세 조정을 마친 LLM을 통해 여러 응답을 생성하고 그중에서 리워드 모델이 가장 높은 점수를 준 응답을 모아 다시 지도 미세 조정을 수행   
강화 학습 이전에 기각 샘플링을 통해 언어 모델이 더 빠르고 안정적으로 사람의 선호를 학습한 후 PPO를 사용   

### 4.3.2 DPO: 선호 데이터셋을 직접 학습하기
**DPO**(Direct Preference Optimization) - 리워드 모델과 강화 학습을 사용하지 않고 선호 데이터셋을 직접 학습   
그림 4.17   
별도로 리워드 모델과 강화학습이 필요없기 때문에 쉽고 빠르게 모델에 사람의 선호를 반영할 수 있음   
선호 데이터셋을 직접보면서 선호도가 높은 데이터의 특성을 익히고 이를 생성하는 방법을 더 직접적으로 학습할 수 있음   

### 4.3.3 DPO를 사용해 학습한 모델들
허깅페이스 팀이 2023년 공개한 제퍼-7B-베타는 AI 평가를 통해 DPO학습 데이터를 구축   
4개의 LLM이 생성한 결과를 AI가 평가하고 이를 선호/비선호 데이터 쌍을 구축해 dDPO를 수행하여 구현   
최고 성능의 사전 학습 모델로 DPO가 성공적으로 동작하고 더 나아가 사람의 평가가 아닌 AI 평가로도 잘 동작한다는 사실을 증명   
이후 뉴럴-챗-7B나 앨런 AI와 같이, 더 큰 모델에서의 확장성에 대한 의문이 줄어들었고 LLM의 선호 학습을 가속화하는 핵심 기술이 됨   

# 5 GPU 효율적인 학습
**GPU**(Graphic Processing Unit) - 단순한 곱셈을 동시에 여러 개 처리하는데 특화된 처리 장치   
딥러닝 모델이 입력 데이터를 처리해 결과를 내 놓을 때까지 많은 행렬 곱셈과 같은 단순한 연산을 처리하는 데 GPU를 활용   

## 5.1 GPU에 올라가는 데이터 살펴보기
**OOM(Out of Memory) 에러** - 한정된 GPU 메모리에 데이터가 가득 차 더 이상 새로운 데이터를 추가하지 못해 발생하는 에러   

### 5.1.1 딥러닝 모델의 데이터 타입
과거에는 딥러닝 모델을 32비트 부동소수점 형식을 사용했으나 점점 더 파라미터가 많은 모델을 사용하면서 최근에는 주로 16비트로 수를 표현하는 fp16 또는 bf16(brain float 16)을 사용   
fp16이 표현할 수 있는 수의 범위가 좁기 때문에 지수에 8비트 가수에 7비트를 사용하는 **bf16**을 개발   
그림 5.1   

### 5.1.2 양자화로 모델 용량 줄이기
**양자화**(quantization) - 기존보다 더 적은 비트로 모델을 표현하는 기술   
원본 데이터의 정보를 최대한 유지하면서 더 적은 용량의 데이터 형식으로 변환하려면, 변환하려는 데이터 형식의 수를 최대한 낭비하지 않고 사용해야 함   
데이터 형식의 최대와 최소를 대응시키면서 간단하게 양자화를 진행하면 양쪽 끝에 사용하는 데이터가 없이 존재해 낭비되는 문제가 발생   
대응시키지 않고 데이터의 최대값 범위로 양자화할 수 있지만 이상치(outlier)가 있는 경우에는 취약   
전체 데이터에 동일한 변환이 아닌 데이터를 묶은 블록 단위 양자화를 수행    
입력 데이터를 크기 순으로 등수를 매겨 배치하는 퀀타일(quantile) 방식     
입력 데이터의 등수를 확인해야 하고 배치해야 하기 때문에 계산량도 많고 별도로 메모리를 사용한다는 단점   

### 5.1.3 GPU 메모리 분해하기
GPU 메모리에는 다음과 같은 데이터가 저장

> **모델 파라미터**   
> **그레이디언트**(gradient)   
> **옵티마이저 상태**(optimizer state)   
> **순전파 상태**(forward activation)   

딥러닝의 과정 - 먼저 순전파를 수행하고 그때 계산한 손실로부터 역전파를 수행하고 마지막으로 옵티마이저를 통해 모델을 업데이트    
모델의 용량이 N일 경우 모델 파라미터(N) + 그레이디언트(N) + 옵티마이저 상태(2N) -> 대략 **4N의 메모리 + 추가로 순전파 상태일 때의 메모리** 또한 필요   
배치 크기가 증가해도 모델, 그레이디언트, 옵티마이저 상태를 저장하는 데 필요한 GPU 메모리는 동일 / 오로지 순전파 상태의 계산에 필요한 메모리가 증가   
표 5.1   

## 5.2 단일 GPU 효율적으로 활용하기
### 5.2.1 그레이디언트 누적
**그레이디언트 누적**(gradient accumulation) - 제한된 메모리 안에서 배치 크기를 키우는 것과 동일한 효과를 얻는 방법   
적은 GPU 메모리로도 더 큰 배치 크기와 같은 효과를 얻을 수 있지만, 추가적인 순전파 및 역전파 연산을 수행하기 때문에 학습 시간이 증가   

### 5.2.2 그레이디언트 체크포인팅
역전파 계산을 위해 순전파의 결과를 저장하고 있어야 하는데 기본적으로는 '모두' 저장함   
이를 절약하기 위해 필요한 최소 데이터만을 저장하고 나머지는 필요할 때 다시 계산하는 방식을 사용   
메모리를 효율적으로 쓸 수 있지만 한 번의 역전파를 위해 순전파를 반복적으로 계산해야 함   
**그레이디언트 체크포인팅**(gradient checkpointing) - 두 방법을 절충해서 중간중간에 값들을 저장해서 메모리 사용을 줄이고 필요한 경우 체크포인트부터 다시 계산해 계산량도 줄인 방법   
그림 5.9   

## 5.3 분산 학습과 ZeRO
### 5.3.1 분산 학습
**분산 학습**(distributed training) - GPU 메모리의 총량을 늘려 2개 이상의 GPU를 사용해 모델을 학습시키는 방법   
**데이터 병렬화**(data parallelism) - 여러 GPU에 각각 모델을 올리고 학습 데이터를 병렬로 처리해 학습 속도를 높일 수 있음   
하나의 GPU에 올리기 어려운 큰 모델의 경우 **모델 병렬화**(model parallelism)을 사용해 여러 개의 GPU에 나눠서 올림   

> 딥러닝 모델의 층별로 나눠 GPU에 올리는 **파이프라인 병렬화**(pipeline parallelism)   
> 한 층의 모델도 나눠서 GPU에 올리는 **텐서 병렬화**(tensor parallelism)   

그림 5.11   
위 그림을 상하로 나누면 파이프라인 병렬화이고, 좌우로 나누면 텐서 병렬화에 해당   
**파이프라인 병렬화**의 경우 딥러닝 모델의 층 순서에 맞춰 순차적으로 연산 vs **텐서 병렬화**의 경우 행렬을 분리해도 동일한 결과를 얻을 수 있도록 행렬 곱셈을 적용   
데이터 병렬화의 경우 동일한 모델을 여러 GPU에 올리기 때문에 중복으로 모델을 저장하면서 메모리 낭비가 발생   

### 5.3.2 데이터 병렬화에서 중복 저장 줄이기(ZeRO)
**ZeRO**(Zero Redundancy Optimizer) - 마이크로소프트에서 개발했고 하나의 모델을 모델 병렬화처럼 여러 GPU에 나눠 올리고 각 GPU에서는 자신의 모델 부분의 연산만 수행하는 컨셉   
그림 5.14

## 5.4 효율적인 학습 방법(PEFT): LoRA
LLM과 같은 모델의 크기가 커지면서 모든 파라미터가 아닌 일부만 학습하는 **PEFT**(Parameter Efficient Fine-Tuning)방법 연구가 활발   
### 5.4.1 모델 파라미터의 일부만 재구성해 학습하는 LoRA
**LoRA**(Low Rank Adaptation) - 모델 파라미터를 재구성(reparameterization)해 더 적은 파라미터를 학습함으로써 GPU 메모리 사용량을 감소   
기존의 파라미터는 고정한 상태에서 새로운 저차원 행렬을 구성하여 추가 학습을 진행하여 개선   
모델 파라미터 용량 자체는 아주 작게 증가하지만 앞서 말한 GPU 메모리 차원에서는 작은 행렬을 계산하기 때문에 훨씬 효율적   
그림 5.16   

### 5.4.2 LoRA 설정 살펴보기
먼저 새로운 저차원 행렬을 만들 때 차원을 몇으로 할지 정해야 함   
차원이 작을 수록 GPU 메모리 사용량을 더 줄일 수 있지만 그만큼 모델이 학습할 수 있는 용량이 작아지기 때문에 데이터의 패턴을 충분히 학습하지 못할 수 있음   
다음으로 추가한 파라미터를 기존 파라미터에 얼마나 많이 반영할지 결정하는 알파 값도 정해야 함   
알파가 커질수록 새롭게 학습한 파라미터의 중요성이 커지기 때문에 적절한 알파 값도 설정해야 함   
마지막으로 어떤 파라미터를 재구성할지 결정해야 함   

### 5.4.3 코드로 LoRA 학습 사용하기
```
cleanup()
print_gpu_utilization()

gpu_memory_experiment(batch_size=16, peft='lora')

torch.cuda.empty_cache()

#출력결과
#배치 사이즈: 16
#GPU 메모리 사용량: 2.618 GB
#GPU 메모리 사용량: 4.732 GB
#옵티마이저 상태의 메모리 사용량: 0.006 GB
#그레디언트 메모리 사용량: 0.003 GB
#GPU 메모리 사용량: 0.016 GB
``` 
LoRA를 적용하니 전체 파라미터 대비 0.117%로 훨씬 줄어들어 학습하기 때문에 옵티마이저 상태의 메모리 사용량과 그레이디언트 사용량이 매우 줄어들었음   

## 5.5 효율적인 학습방법(PEFT): QLoRA
**QLoRA** - 기존의 LoRA 방식에 양자화를 추가해 메모리 효율성을 한 번 더 높인 학습 방법   
### 5.5.1 4비트 양자화와 2차 양자화
입력이 정규 분포라는 가정을 활용하여 모델의 성능을 거의 유지하면서도 빠른 양자화가 가능해짐 - 4비트 부동소수점 데이터 형식인 **NF4**(Normal Float 4-bit)를 제안   
**2차 양자화** - NF4 양자화 과정에서 생기는 32비트 상수를 효율적으로 저장하는 방법   

### 5.5.2 페이지 옵티마이저
**페이지 옵티마이저**(paged optimizer) - 엔비디아의 통합 메모리를 통해 GPU가 CPU 메모리(RAM)를 공유하는 것   
컴퓨터의 가상 메모리 시스템과 유사한 개념을 GPU 메모리 관리에 적용한 기술   
가상 메모리에서 운영체제는 램이 가득 차면 일부 데이터를 디스크로 옮기고 필요할 때 다시 램으로 데이터를 불러옴 - 페이징(paging)   

### 5.5.3 코드로 QLoRA 모델 활용하기
```
cleanup()
print_gpu_utilization()

gpu_memory_experiment(batch_size=16, peft='qlora')

torch.cuda.empty_cache()

#출력결과
#GPU 메모리 사용량: 0.945 GB
#배치 사이즈: 16
#GPU 메모리 사용량: 2.112 GB
#GPU 메모리 사용량: 2.651 GB
#옵티마이저 상태의 메모리 사용량: 0.012 GB
#그레디언트 메모리 사용량: 0.006 GB
#GPU 메모리 사용량: 0.945 GB 
```  
QLoRA를 사용할 경우 메모리 사용량이 절반 이하로 떨어짐  

# 6 sLLM 학습하기
**Text2SQL** - 사용자가 얻고 싶은 데이터에 대한 요청을 자연어로 작성하면 LLM이 요청에 맞는 SQL을 생성하는 작업   
## 6.1 Text2SQL 데이터셋
### 6.1.1 대표적인 Text2SQL 데이터셋
대표적인 Text2SQL 데이터셋으로는 WikiSQL과 Spider가 존재   
SQL을 생성하기 위해서는 두 가지 데이터가 필요

> 어떤 데이터가 있는지 알 수 있는 데이터베이스 정보(테이블과 칼럼)
> 어떤 데이터를 추출하고 싶은지 나타낸 요청사항(request/question)   

### 6.1.2 한국어 데이터셋
현재는 데이터 보완 작업을 위해 공개가 중단돼 있지만 AI 허브에서 구축한 데이터셋은 모델학습을 위한 목적으로만 활용될 수 있음

### 6.1.3 합성 데이터 활용
그림 6.2   
그림과 같이 db_id, context, question, answer 4개의 컬럼으로 구성되어 있음   

> db_id는 테이블이 포함된 데이터베이스의 아이디로 동일한 값을 갖는 테이블은 같은 도메인을 공유   
> context 컬럼은 SQL 생성에 사용할 테이블 정보를 가지고 있음    
> question과 answer은 각각 데이터 요청사항, 요청에 대한 SQL 정답을 담고 있음   

## 6.2 성능 평가 파이프라인 준비하기
### 6.2.1 Text2SQL 평가 방식
**EM(Exact Match)방식** - 생성한 SQL이 문자열 그대로 동일한지 확인하는 방법   
**실행 정확도**(Execution Accuracy) - SQL 쿼리를 수행해 정답과 일치하는지 확인하는 방식   
EM 방식은 문자열이 완전히 동일하지 않으면 다르다고 판단한다는 문제가 있음   
실행 정확도의 경우 쿼리를 실행할 수 있는 데이터베이스를 추가로 준비해야 함   
그림 6.5   
GPT를 활용한 성능 평가 파이프라인을 준비하기 위해서 필요한 세가지   

> 1. 먼저 평가 데이터셋을 구축   
> 2. 다음으로 LLM이 SQL을 생성할 때 사용할 프롬프트를 준비
> 3. GPT 평가에 사용할 프롬프트와 GPT-4 API 요청을 빠르게 수행할 수 있는 코드를 작성

### 6.2.2 평가 데이터셋 구축
합성 데이터셋 전부를 직접 검수하기에는 용량 문제로 인해서 선별된 데이터셋만을 활용   

### 6.2.3 SQL 생성 프롬프트
LLM의 경우 학습에 사용한 프롬프트 형식을 추론할 때도 동일하게 사용해야 결과 품질이 좋기 때문에 지시사항과 데이터로 나눈 프롬프트를 동일하게 사용   
```
def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
{ddl}

### Question:
{question}

### SQL:
{query}"""
    return prompt
```

### 6.2.4 GPT-4 평가 프롬프트와 코드 준비
GPT-4를 사용해 평가를 수행해야 하기 때문에 반복적으로 API를 요청해야 함   
jsonl 파일을 작성해 요청 제한을 관리하면서 비동기적으로 요청을 보낼 수 있음   
OpenAI는 사용자에 따라 티어를 나누어 사용량 제한에 차등을 두고 있음   
그림 6.7   

## 6.3 실습: 미세 조정 수행하기
### 6.3.1 기초 모델 평가하기
```
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def make_inference_pipeline(model_id):
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  return pipe

model_id = 'beomi/Yi-Ko-6B'
hf_pipe = make_inference_pipeline(model_id)

example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
CREATE TABLE players (
  player_id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(255) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  date_joined DATETIME NOT NULL,
  last_login DATETIME
);

### Question:
사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

### SQL:
"""

hf_pipe(example, do_sample=False,
    return_full_text=False, max_length=512, truncation=True)
#  SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';

# ### SQL 봇:
# SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';

# ### SQL 봇의 결과:
# SELECT COUNT(*) FROM players WHERE username LIKE '%admin%'; (생략)
```  
위의 코드를 실행하면 요청에 맞춰 SQL은 잘 생성되었지만 반복적으로 'SQL 봇', 'SQL 봇의 결과'와 같이 추가적인 결과를 생성   
형식에 맞춰 선호도가 높은 답변을 얻기 위해서는 추가적인 학습이 필요   

### 6.3.2 미세 조정 수행   
모델 학습 과정에서 메모리 에러가 발생할 경우 batch_size를 줄여서 다시 시도   
예제의 경우 구글 코랩 프로의 A100 GPU 기준으로 약 1시간 소요 / T4 GPU의 경우 8~10배의 시간 더 소요
표 6.1   

### 6.3.3 학습 데이터 정제와 미세 조정
이전 예제의 학습 데이터는 GPT가 생성한 원본 데이터이기 때문에 잘못 생성된 데이터가 상당 부분 존재   
다시 GPT를 활용해 SQL을 맥락과 요청에 따라 필터링을 진행하여 데이터셋의 크기가 1/4 정도 줄어들었음에도 정제 전의 데이터셋으로 학습했을 때와 동일한 성능 달성   
표 6.2   
성능이 동일한 것이 데이터셋의 크기가 달라져도 성능에 영향이 없는지 확인   

### 6.3.4 기초 모델 변경
기존에 사용하던 beomi/Yi-KO-6B보다 두 배 더 크고 성능 면에서 뛰어난 beomi/OPEN-SOLAR-KO-10.7B 모델을 사용하여 예제를 수행   
미세 조정 전과 후 모두 기존 모델에 비해 월등히 높은 성능을 달성   
표 6.4   

### 6.3.5 모델 성능 비교
그림 6.8   
기초 모델이 GPT에 비하면 매우 작은 모델이고 A100 GPU 기준 1시간정도 학습한 것을 고려했을 때 충분히 인상적인 결과를 보여줌   
