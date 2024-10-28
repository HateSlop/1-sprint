# Week2

LLM을 활용한 실전 AI 애플리케이션 개발

## 07 모델 가볍게 만들기

```python
!pip install transformers==4.40.1 accelerate==0.30.0 bitsandbytes==0.43.1 auto-gptq==0.7.1 autoawq==0.2.5 optimum==1.19.1 -qqq
```

### 7.1 언어 모델 추론 이해하기

7.1.1 언어 모델이 언어를 생성하는 방법 

Auto-regressive

7.1.2 중복 연산을 줄이는 KV 캐시

7.1.3 GPU 구조와 최적의 배치 크기

7.1.4 KV 캐시 메모리 줄이기

- 멀티 쿼리 어텐션
- 그룹 쿼리 어텐션
- 멀티 헤드 어텐션

### 7.2 양자화로 모델 용량 줄이기

- 학습 후 양자화 e.g. 비츠앤바이츠, GPTQ, AWQ
- 양자화 학습

7.2.1 비츠앤 바이츠

```python
# 비츠앤바이츠 양자화 모델 불러오기 

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8비트 양자화 모델 불러오기
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=bnb_config_8bit)

# 4비트 양자화 모델 불러오기
bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4")

model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m",
                                                  low_cpu_mem_usage=True,
                                                  quantization_config=bnb_config_4bit)
```

7.2.2 GPTQ

```python
# GPTQ 양자화 수행 코드
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

# GPTQ 양자화된 모델 불러오기
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ",
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
```

7.2.3 AWQ

```python
# AWQ 양자화 모델 불러오기
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True)
```

### 7.3 지식 증류 활용하기

**지식 증류**: 더 크고 성능이 높은 선생 모델의 생성 결과를 활용해 더 작고 성능이 낮은 학생 모델을 만드는 방법

### 7.4 정리

양자화와 지식 증류는 성능의 희생을 감수하여 추론을 효율화 하는 방식

## 08 sLLM 서빙하기

```jsx
!pip install transformers==4.40.1 accelerate==0.30.0 bitsandbytes==0.43.1 datasets==2.19.0 vllm==0.4.1 openai==1.25.1 -qqq
```

### 8.1 효율적인 배치 전략

8.1.1 일반 배치(정적 배치)

8.1.2 동적 배치

8.1.3 연속 배치

### 8.2 효율적인 트랜스포머 연산

8.2.1 플래시 어텐션

8.2.2 플래시 어텐션 2

- 행렬 곱셈이 아닌 연산 줄이기
- 시퀀스 길이 방향의 병렬화 추가

8.2.3 상대적 위치 인코딩

### 8.3 효율적인 추론 전략

8.3.1 커널 퓨전

8.3.2 페이지어텐션

8.3.3 추측 디코딩

### 8.4 실습: LLM 서빙 프레임워크

8.4.1 오프라인 서빙

8.4.2 온라인 서빙

### 8.5 정리

## 09 LLM 어플리케이션 개발하기

환각(Hallucination)

```jsx
!pip install datasets llama-index==0.10.34 langchain-openai==0.1.6 "nemoguardrails[openai]==0.8.0" openai==1.25.1 chromadb==0.5.0 wandb==0.16.6 -qqq
!pip install llama-index-callbacks-wandb==0.1.2 -qqq
```

### 9.1 검색 증강 생성(RAG)

9.1.1 데이터 저장

9.1.2 프롬프트에 검색 결과 통합

9.1.3 실습: 라마인덱스로 RAG 구현하기

```python
import os
from datasets import load_dataset

os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"

dataset = load_dataset('klue', 'mrc', split='train')
dataset[0]
```

```python
from llama_index.core import Document, VectorStoreIndex

text_list = dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

# 인덱스 만들기
index = VectorStoreIndex.from_documents(documents)
```

```python
print(dataset[0]['question']) # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?

retrieval_engine = index.as_retriever(similarity_top_k=5, verbose=True)
response = retrieval_engine.retrieve(
    dataset[0]['question']
)
print(len(response)) # 출력 결과: 5
print(response[0].node.text)
```

```python
query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query(
    dataset[0]['question']
)
print(response)
# 장마전선에서 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 한 달 정도입니다.
```

```python
from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# 검색을 위한 Retriever 생성
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=1,
)

# 검색 결과를 질문과 결합하는 synthesizer
response_synthesizer = get_response_synthesizer()

# 위의 두 요소를 결합해 쿼리 엔진 생성
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# RAG 수행
response = query_engine.query("북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?")
print(response)
# 장마전선에서 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 한 달 가량입니다.
```

### 9.2 LLM 캐시

9.2.1 LLM 캐시의 작동 원리

9.2.2 실습: OpenAI API 캐시 구현

### 9.3 데이터 검증

9.3.1 데이터 검증 방식

- 규칙 기반
- 분류 또는 회귀 모델
- 임베딩 유사도 기반
- LLM 활용

9.3.2 데이터 검증 실습

### 9.4 데이터 로깅

### 9.5 정리

## 10 임베딩 모델로 데이터 의미 압축하기

```python
!pip install transformers==4.40.1 datasets==2.19.0 sentence-transformers==2.7.0 faiss-cpu==1.8.0 llama-index==0.10.34 llama-index-embeddings-huggingface==0.2.0 -qqq
```

### 10.1 텍스트 임베딩 이해하기

10.1.1 문장 임베딩 방식의 장점

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
dense_embeddings = smodel.encode(['학교', '공부', '운동'])
cosine_similarity(dense_embeddings) # 코사인 유사도
# array([[1.0000001 , 0.5950744 , 0.32537547],
#       [0.5950744 , 1.0000002 , 0.54595673],
#       [0.32537547, 0.54595673, 0.99999976]], dtype=float32)
```

10.1.2 원핫 인코딩

```python
# 원핫 인코딩의 한계
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

word_dict = {"school": np.array([[1, 0, 0]]),
"study": np.array([[0, 1, 0]]),
"workout": np.array([[0, 0, 1]])
}

# 두 단어 사이의 코사인 유사도 계산하기
cosine_school_study = cosine_similarity(word_dict["school"], word_dict['study']) # 0
cosine_school_workout = cosine_similarity(word_dict['school'], word_dict['workout']) # 0
```

10.1.3 백오브워즈 

10.1.4 TF-IDF

10.1.5 워드투벡(word2vec)

### 10.2 문장 임베딩 방식

10.2.1 문장 사이의 관계를 계산하는 두 가지 방법

10.2.2 바이 인코더 모델 구조

10.2.3 Sentence-Transformers로 텍스트와 이미지 임베딩 생성해 보기

10.2.4 오픈소스와 상업용 임베딩 모델 비교하기

### 10.3 실습: 의미 검색 구현하기

10.3.1 의미 검색 구현하기

10.3.2 라마인덱스에서 Sentence-Transformers 모델 사용하기

### 10.4 검색 방식을 조합해 성능 높이기

10.4.1 키워드 검색 방식: BM25

10.4.2 상호 순위 조합 이해하기

### 10.5 실습: 하이브리드 검색 구현하기

10.5.1 BM25 구현하기

10.5.2 상호 순위 조합 구현하기

10.5.3 하이브리드 검색 구현하기 

### 10.6 정리

## 11 자신의 데이터에 맞춘 임베딩 모델 만들기: RAG 개선하기

```python
!pip install sentence-transformer==2.7.0 datasets==2.19.0 huggingface_hub ==0.23.0 faiss-cpu==1.8.0 -qqq
```

### 11.1 검색 성능을 높이기 위한 두 가지 방법

|  | 장점 | 단점 |
| --- | --- | --- |
| Bi-encoder | 가벼운 벡터 연산 | 정확하지 못한 유사도 계산 |
| Cross_encoder | 정확한 유사도 검색 | 계산 복잡도로 인한 확장성 부족 |

### 11.2 언어 모델을 임베딩 모델로 만들기

- 대조학습
- 실습: 학습 준비하기

https://blog.selectstar.ai/ko/klue-problem-solving/

```python
# 사전 학습된 언어 모델을 불러와 문장 임베딩 모델 만들기
# klue/roberta-base : 한국어 벤치마크 데이터셋
from sentence_transformers import SentenceTransformer, models
transformer_model = models.Transformer('klue/roberta-base')

pooling_layer = models.Pooling(
    transformer_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)
embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer]
```

```python
# 실습 데이터셋 다운로드 및 확인 및 검증 데이터셋 분리 
# STS(Sentence Textual Similarity) 2개의 문장이 얼마나 유사한지 점수를 매긴 데이터셋 
from datasets import load_dataset
klue_sts_train = load_dataset('klue', 'sts', split='train')
klue_sts_test = load_dataset('klue', 'sts', split='validation')

# 학습 데이터셋의 10%를 검증 데이터셋으로 구성한다.
klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42)
klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test']

klue_sts_train[0] 

# {'guid': 'klue-sts-v1_train_00000',
#  'source': 'airbnb-rtt',
#  'sentence1': '숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.',
#  'sentence2': '숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.',
#  'labels': {'label': 3.7, 'real-label': 3.714285714285714, 'binary-label': 1}
```

```python
# 라벨 정규화 하기

from sentence_transformers import InputExample
# 유사도 점수를 0~1 사이로 정규화 하고 InputExample 객체에 담는다.
def prepare_sts_examples(dataset):
    examples = []
    for data in dataset:
        examples.append(
            InputExample(
                texts=[data['sentence1'], data['sentence2']],
                label=data['labels']['label'] / 5.0)
            )
    return examples

train_examples = prepare_sts_examples(klue_sts_train)
eval_examples = prepare_sts_examples(klue_sts_eval)
test_examples = prepare_sts_examples(klue_sts_test)
```

```python
# 학습에 사용할 배치 데이터셋 만들기
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 검증을 위한 평가 객체 준비
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

# 언어 모델을 그대로 활용할 경우 문장 임베딩 모델의 성능
test_evaluator(embedding_model)
# 0.36460670798564826
```

- 실습: 유사한 문장 데이터로 임베딩 모델 학습하기

```python
# 임베딩 모델 학습 

from sentence_transformers import losses

num_epochs = 4
model_name = 'klue/roberta-base'
model_save_path = 'output/training_sts_' + model_name.replace("/", "-")
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

# 임베딩 모델 학습
embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=eval_evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=100,
    output_path=model_save_path
)
```

```python
# 학습된 임베딩 모델의 성능 평가

trained_embedding_model = SentenceTransformer(model_save_path)
test_evaluator(trained_embedding_model)
# 0.8965595666246748
```

```python
# 허깅페이스 허브에 모델 저장

from huggingface_hub import login
from huggingface_hub import HfApi

login(token='허깅페이스 허브 토큰 입력')
api = HfApi()
repo_id="klue-roberta-base-klue-sts"
api.create_repo(repo_id=repo_id)

api.upload_folder(
    folder_path=model_save_path,
    repo_id=f"본인의 허깅페이스 아이디 입력/{repo_id}",
    repo_type="model",
)
```

### 11.3 임베딩 모델 미세 조정하기

```python
# 실습 데이터를 내려받고 예시 데이터 확인
# MRC: 기사 본문 및 해당 기사와 관련된 질문을 수집한 데이터셋

from datasets import load_dataset
klue_mrc_train = load_dataset('klue', 'mrc', split='train')
klue_mrc_train[0]

# {'title': '제주도 장마 시작 … 중부는 이달 말부터',
#  'context': '올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 이달 말께 장마가 시작될 전망이다.17일 기상청에 따르면 제주도 남쪽 먼바다에 있는 장마전선의 영향으로 이날 제주도 산간 및 내륙지역에 호우주의보가 내려지면서 곳곳에 100㎜에 육박하는 많은 비가 내렸다. 제주의 장마는 평년보다 2~3일, 지난해보다는 하루 일찍 시작됐다. 장마는 고온다습한 북태평양 기단과 한랭 습윤한 오호츠크해 기단이 만나 형성되는 장마전선에서 내리는 비를 뜻한다.장마전선은 18일 제주도 먼 남쪽 해상으로 내려갔다가 20일께 다시 북상해 전남 남해안까지 영향을 줄 것으로 보인다. 이에 따라 20~21일 남부지방에도 예년보다 사흘 정도 장마가 일찍 찾아올 전망이다. 그러나 장마전선을 밀어올리는 북태평양 고기압 세력이 약해 서울 등 중부지방은 평년보다 사나흘가량 늦은 이달 말부터 장마가 시작될 것이라는 게 기상청의 설명이다. 장마전선은 이후 한 달가량 한반도 중남부를 오르내리며 곳곳에 비를 뿌릴 전망이다. 최근 30년간 평균치에 따르면 중부지방의 장마 시작일은 6월24~25일이었으며 장마기간은 32일, 강수일수는 17.2일이었다.기상청은 올해 장마기간의 평균 강수량이 350~400㎜로 평년과 비슷하거나 적을 것으로 내다봤다. 브라질 월드컵 한국과 러시아의 경기가 열리는 18일 오전 서울은 대체로 구름이 많이 끼지만 비는 오지 않을 것으로 예상돼 거리 응원에는 지장이 없을 전망이다.',
#  'news_category': '종합',
#  'source': 'hankyung',
#  'guid': 'klue-mrc-v1_train_12759',
#  'is_impossible': False,
#  'question_type': 1,
#  'question': '북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?',
#  'answers': {'answer_start': [478, 478], 'text': ['한 달가량', '한 달']}}
```

```python
# 기본 임베딩 모델 불러오기
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('shangrilar/klue-roberta-base-klue-sts')
```

```python
# 데이터 전처리

from datasets import load_dataset
klue_mrc_train = load_dataset('klue', 'mrc', split='train')
klue_mrc_test = load_dataset('klue', 'mrc', split='validation')

df_train = klue_mrc_train.to_pandas()
df_test = klue_mrc_test.to_pandas()

df_train = df_train[['title', 'question', 'context']]
df_test = df_test[['title', 'question', 'context']]
```

```python
# 질문과 관련이 없는 기사를 irrelavant_contexts 칼럼에 추가
def add_ir_context(df):
  irrelevant_contexts = []
  for idx, row in df.iterrows():
    title = row['title']
    irrelevant_contexts.append(df.query(f"title != '{title}'").sample(n=1)['context'].values[0])
  df['irrelevant_context'] = irrelevant_contexts
  return df

df_train_ir = add_ir_context(df_train)
df_test_ir = add_ir_context(df_test)
```

```python
# 성능 평가에 사용할 데이터 생성

from sentence_transformers import InputExample

examples = []
for idx, row in df_test_ir[:100].iterrows():
  examples.append(
      InputExample(texts=[row['question'], row['context']], label=1)
  )
  examples.append(
      InputExample(texts=[row['question'], row['irrelevant_context']], label=0)
  )
```

```python
# 기본 임베딩 모델의 성능 평가 결과

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    examples
)
evaluator(sentence_model)
# 0.8151553052035344
```

### 11.4 검색 품질을 높이는 순위 재정렬

### 11.5 바이 인코더와 교차 인코더로 개선된 RAG 구현하기

1. 기본 임베딩 모델로 검색하기
2. 미세 조정한 임베딩 모델로 검색하기
3. 미세 조정한 모델과 교차 인코더를 결합해 검색하기 

### 11.6 정리

- RAG 성능 향상을 위해 바이인코더와 교차인코더를 함께 활용