# LLM을 활용한 실전 AI 애플리케이션 개발
------------------
# 12 벡터 데이터베이스로 확장하기: RAG 구현하기
## 12.1 벡터 데이터베이스란
**벡터 데이터베이스** - 벡터 임베딩(데이터의 의미를 담은 숫자 배열)을 키로 사용하는 데이터베이스   

### 12.1.1 딥러닝과 벡터 데이터베이스
**표현 학습** - 기존의 머신러닝에서는 매번 새롭게 특징을 정의하는 단계가 필요했지만 딥러닝에서는 데이터만 충분하다면 모델이 알아서 이런 특징을 뽑는 과정도 학습   

그림 12.4   

비슷한 데이터는 가깝게 있고, 다른 데이터는 멀리 위치하게 되는 임베딩 벡터의 특징을 이용해 서로 비슷한 데이터를 찾을 수 있음   
벡터 데이터 베이스를 활용하기 위한 3단계   
1. 저장: 저장할 데이터를 임베딩 모델을 거쳐 벡터로 변환하고 벡터 데이터베이스에 저장
2. 검색: 검색할 데이터를 임베딩 모델을 거쳐 벡터로 변환하고 벡터 데이터베이스에서 검색
3. 결과 반환: 벡터 데이터베이스에서는 검색 쿼리의 임베딩과 거리가 가까운 벡터를 찾아 반환

벡터 사이의 거리를 측정하는 다양한 방법이 있는데 일반적으로 유클리드 거리, 코사인 유사도, 점곱(dot product)을 가장 많이 활용   
### 12.1.2 벡터 데이터베이스 지형 파악하기
벡터 임베딩을 저장하고 검색하는 기능을 구현할 때 크게 아래와 같은 소프트웨어를 접함   
1. **벡터 라이브러리** - 메타의 Faiss, 스포티파이의 Annoy 와 같이 벡터를 저장하고 검색하는 핵심 기능을 구현   
2. **벡터 전용 데이터베이스** - 파인콘(Pinecone), 위비에이트(Weaviate)
3. **벡터 기능 추가 데이터베이스** - 일래스틱서치(Elasticsearch), PostgreSQL과 같이 기존의 데이터베이스에 벡터 저장과 검색 기능을 추가   

벡터 데이터베이스는 벡터 라이브러리와 다르게 아래와 같은 기능을 제공   
1. 메타 데이터의 저장 및 필터링 가능
2. 데이터의 백업 및 관리
3. 모니터링, 관련 AI 도구 등 에코시스템과의 통합
4. 데이터 보안과 엑세스 관리   

그림 12.7

고급 벡터 검색이 필요하고 워크로드가 큰 경우 그림 왼쪽의 벡터 전용 데이터베이스를 선택하는 것이 좋음   
벡터 데이터베이스에 대한 이해도가 있고 직접 오픈소스 서비스를 활용해 시스템을 구축할 수 있고 선호한다면 그림 왼쪽 위의 오픈소스 벡터 데이터베이스가 좋음   

## 12.2 벡터 데이터베이스 작동 원리
### 12.2.1 KNN 검색과 그 한계
**KNN(K-Nearest Neighbor)검색** - 검색하려는 벡터와 가장 가까운 K개 이웃 벡터를 찾는 검색 방식   
모든 데이터를 조사하기 때문에 정확하지만 모든 벡터를 조사하기 때문에 연산량이 데이터 수에 비례하게 늘어남   
벡터 검색을 위해서는 먼저 **인덱스**(관계형 데이터베이스의 테이블과 비슷한 레벨)를 만들어야 함   
인덱스를 벡터에 저장하는데, 이렇게 벡터를 저장하는 과정을 "**색인한다**"라고 함   
색인 단계에서는 인덱스의 메모리 사용량과 색인 시간이 중요하고 검색 단계에서는 검색 시간과 재현율이 중요   
**재현율** - 실제로 가장 가까운 K개의 정답 데이터 중 몇 개가 검색 결과로 반환됐는지 그 비율을 나타낸 값   
KKN 검색의 경우 재현율이 100%   

그림 12.8   

### 12.2.2 ANN 검색이란
**근사 최근접 이웃(Approximate Nearest Neighbor)검색** - 대용량 데이터셋에서 주어진 쿼리 항목과 가장 유사한 항목을 효율적으로 찾는 데 사용되는 기술   
대표적인 ANN 알고리즘   
1. **IVF(Inverted File Index)** - 검색 공간을 제한하기 위해 데이터셋 벡터들을 클러스터로 그룹화
2. **HNSW(Hierarchical Navigable Small World)** - 효율적인 ANN 검색을 위한 그래프 기반 인덱싱 구조   

HNSW가 가장 많이 활용되는 ANN 검색 알고리즘   
```
ANN 검색의 재현율 = (KNN으로 찾은 실제 가장 가까운 K개 중 ANN이 찾은 개수) / K
```

### 12.2.3 탐색 가능한 작은 세계(NSW)
HNSW의 그래프는 노드(node)와 간선(edge)로 이루어짐   
1. 노드 - 저장하는 데이터를 의미하고 벡터 데이터베이스에서는 벡터 임베딩이 노드
2. 간선 - 노드와 노드를 연결하는 선으로, 간선을 통해 서로 연결된 노드끼리만 탐색이 가능   

그림 12.9   

탐색 가능한 작은 세계 - 완전히 랜덤한 그래프와 완전히 규칙적인 그래프 사이에 '적당히 랜덤하게' 연결된 그래프 상태   
규칙적인 연결을 통해 정확한 탐색이 가능하면서도 랜덤한 성질을 통해 빠른 탐색이 가능   

그림 12.10   

하지만 랜덤으로 저장하다 보니 아래 그림과 같이 진입점에서 출발했을 때 찾으려는 검색 벡터(Q)와 가장 가까운 점(E)이 아닌 점 A에서 탐색을 멈추는 **지역 최솟값(local mininum)** 문제 발생   

그림 12.13   

### 12.2.4 계층 구조
**연결 리스트(linked list)** - 새로운 데이터를 추가하거나 삭제할 때 서로를 연결하는 주소 정보를 추가하거나 삭제하면 되기 때문에 데이터 추가/삭제가 자유롭지만 탐색을 할 때는 앞에서부터 순차적으로 확인해야 하기 때문에 탐색 속도가 느림   

그림 12.14   

아래 그림과 같이 데이터가 크기 순으로 정렬되어 있다면 레벨을 나누어 데이터를 듬성듬성 배치하고 탐색은 가장 위층부터 시작   

그림 12.15   

HNSW는 이런 계층 구조를 NSW에 접목해 벡터를 저장하는데 아래와 같은 기준으로 벡터를 저장   
1. 최대가 6인 주사위를 굴려서 6이 나오면 0,1,2층 모두 배치   
2. 주사위를 굴려서 4~5가 나오면 0,1층에 배치
3. 주사위를 굴려서 1,2,3이 나오면 0층에만 배치   

그림 12.16   

## 12.3 실습: HNSW 인덱스의 핵심 파라미터 이해하기
### 12.3.1 파라미터 m 이해하기
HNSW에서 파라미터 m은 추가하는 임베딩 벡터에 연결하는 간선의 수   
벡터에 연결되는 간선이 많을수록 그래프가 더 촘촘하게 연결되기 때문에 검색의 품질이 좋음   
```
import numpy as np

k=1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for m in [8, 16, 32, 64]:
    index = faiss.IndexHNSWFlat(d, m)
    time.sleep(3)
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(f"M: {m} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
```
표 12.2   

### 12.3.2 파라미터 ef_construction 이해하기
**ef_construction**은 M개의 가장 가까운 벡터를 선택할 후보군의 크기로, 이 값이 크면 더 많은 후보를 탐색하기 때문에 실제로 추가한 벡터와 가장 가까운 벡터를 선택할 가능성이 높음   
```
k=1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

for ef_construction in [40, 80, 160, 320]:
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = ef_construction
    time.sleep(3)
    start_memory = get_memory_usage_mb()
    start_index = time.time()
    index.add(xb)
    end_memory = get_memory_usage_mb()
    end_index = time.time()
    print(f"efConstruction: {ef_construction} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")
```
표 12.3   

### 12.3.3 파라미터 ef_search 이해하기
**ef_search**는 ef_construction이 색인 단계에서 후보군의 크기를 결정한 것과 동일하게 검색 단계에서 후보군의 크기를 결정   
```
for ef_search in [16, 32, 64, 128]:
    index.hnsw.efSearch = ef_search
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
    print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}") 
```
표 12.4   

## 12.4 실습: 파인콘으로 벡터 검색 구현하기
### 12.4.1 파인콘 클라이언트 사용법
```
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
# 임베딩 모델 불러오기
sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
# 데이터셋 불러오기
klue_dp_train = load_dataset('klue', 'dp', split='train[:100]')

embeddings = sentence_model.encode(klue_dp_train['sentence'])
```
파인콘 인덱스에 저장할 수 있도록 tolist() 메서드를 사용해 형태를 변경   
```
# 파이썬 기본 데이터 타입으로 변경
embeddings = embeddings.tolist()
# {"id": 문서 ID(str), "values": 벡터 임베딩(List[float]), "metadata": 메타 데이터(dict) ) 형태로 데이터 준비
insert_data = []
for idx, (embedding, text) in enumerate(zip(embeddings, klue_dp_train['sentence'])):
  insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})
```
```
query_response = index.query(
    namespace='llm-book-sub', # 검색할 네임스페이스
    top_k=10, # 몇 개의 결과를 반환할지
    include_values=True, # 벡터 임베딩 반환 여부
    include_metadata=True, # 메타 데이터 반환 여부
    vector=embeddings[0] # 검색할 벡터 임베딩
)
query_response
```

### 12.4.2 라마인덱스에서 벡터 데이터베이스 변경하기
```
# 파인콘 기본 설정
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=pinecone_api_key)
pc.create_index(
    "quickstart", dimension=1536, metric="euclidean", spec=ServerlessSpec("aws", "us-east-1")
)
pinecone_index = pc.Index("quickstart")

# 라마인덱스에 파인콘 인덱스 연결
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```
## 12.5 실습: 파인콘을 활용해 멀티 모달 검색 구현하기
### 12.5.1 데이터셋
```
from datasets import load_dataset

dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')

example_index = 867
original_image = dataset[example_index]['image']
original_prompt = dataset[example_index]['prompt']
print(original_prompt)

# cute fluffy baby cat rabbit lion hybrid mixed creature character concept,
# with long flowing mane blowing in the wind, long peacock feather tail,
# wearing headdress of tribal peacock feathers and flowers, detailed painting,
# renaissance, 4 k
```

### 12.5.2 실습 흐름
1. 원본 이미지와 세 가지 프롬프트로 생성한 3개의 합성 이미지를 비교   
2. 원본 이미지에 대응되는 '원본 프롬프트'를 입력   
3. 전체 프롬프트 텍스트를 텍스트 임베딩 모델로 저장한 벡터 데이터베이스에 원본 이미지를 이미지 임베딩 모델로 변환한 이미지 임베딩으로 검색해 찾은 '유사 프롬프트'를 사용해 이미지를 생성   

그림 12.19   

### 12.5.3 GPT-4o로 이미지 설명 생성하기
```
import requests
import base64
from io import BytesIO

def make_base64(image):
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
  return img_str

def generate_description_from_image_gpt4(prompt, image64):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {client.api_key}"
  }
  payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{image64}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
  }
  response_oai = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  result = response_oai.json()['choices'][0]['message']['content']
  return result
```

### 12.5.4 프롬프트 저장
아래 예제를 통해 프롬프트 임베딩을 저장하고 이미지 임베딩으로 검색할 인덱스를 생성   
```
print(pc.list_indexes())

index_name = "llm-multimodal"
try:
  pc.create_index(
    name=index_name,
    dimension=512,
    metric="cosine",
    spec=ServerlessSpec(
      "aws", "us-east-1"
    )
  )
  print(pc.list_indexes())
except:
  print("Index already exists")
index = pc.Index(index_name)
```
아래 예제를 사용해 생성한 임베딩 벡터를 벡터 데이터베이스에 저장   
```
input_data = []
for id_int, emb, prompt in zip(range(0, len(dataset)), text_embs.tolist(), dataset['prompt']):
  input_data.append(
      {
          "id": str(id_int),
          "values": emb,
          "metadata": {
              "prompt": prompt
          }
      }
  )

index.upsert(
  vectors=input_data
)
```

### 12.5.5 이미지 임베딩 검색
```
from transformers import AutoProcessor, CLIPVisionModelWithProjection

vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(images=original_image, return_tensors="pt")

outputs = vision_model(**inputs)
image_embeds = outputs.image_embeds

search_results = index.query(
  vector=image_embeds[0].tolist(),
  top_k=3,
  include_values=False,
  include_metadata=True
)
searched_idx = int(search_results['matches'][0]['id'])
``` 

### 12.5.6 DALL-E 3로 이미지 생성
아래의 코드를 사용해 3개의 프롬프트에 대한 이미지를 생성
1. GPT-4o가 원본 이미지를 설명해서 작성한 GPT 설명 프롬프트로 이미지 생성
2. 원본 프롬프트를 사용해 이미지 생성
3. 이미지 임베딩으로 검색한 유사 프롬프트를 사용해 이미지 생성   
```
# GPT-4o가 만든 프롬프트로 이미지 생성
gpt_described_image_url = generate_image_dalle3(described_result)
gpt4o_prompt_image = get_generated_image(gpt_described_image_url)
gpt4o_prompt_image
# 원본 프롬프트로 이미지 생성
original_prompt_image_url = generate_image_dalle3(original_prompt)
original_prompt_image = get_generated_image(original_prompt_image_url)
original_prompt_image
# 이미지 임베딩으로 검색한 유사 프롬프트로 이미지 생성
searched_prompt_image_url = generate_image_dalle3(dataset[searched_idx]['prompt'])
searched_prompt_image = get_generated_image(searched_prompt_image_url)
searched_prompt_image
```

출력결과는 아래와 같이 이미지 생성 파라미터에 따라 생성 결과가 달라지고 랜덤성이 있음   

그림 12.20   

# 13 LLM 운영하기
## 13.1 MLOps
**MLOps(Machine Learning Operations)** - 데브옵스(DevOps)의 개념을 머신러닝과 데이터 과학 분야로 확장한 방법론으로 데이터 수집, 전처리, 모델 학습, 평가, 배포, 모니터링 등 머신러닝 프로젝트의 전 과정을 자동화하고 효율화하는 것   

그림 13.1

MLOps는 특히 이전에 수행된 ML 워크플로를 그대로 반복했을 때 동일한 모델을 얻을 수 있는지 여부를 의미하는 **재현성(reproducibility)**를 보장하는 것이 매우 중요   

그림 13.2

### 13.1.1 데이터 관리
모델 학습을 위한 데이터 준비 과정에는 여러가지 중요한 의사결정이 포함되고 그 의사 결정에 따라 다양한 형태의 데이터셋이 생성   
포함시킬 데이터의 범위를 선택하고 어떤 전처리 방식을 포함시킬지, 특성 공학을 통해 어떤 특성을 추가할지에 따라 학습 데이터셋이 달라짐   

그림 13.3   

모델 학습 결과를 재현하기 위해서는 데이터셋의 버전을 관리하고 어떤 학습 데이터셋으로 모델을 학습시켰는지 기록해야 함   

### 13.1.2 실험 관리
머신러닝 모델을 학습시킬 때는 어떤 모델을 사용할지 결정해야 함   

그림 13.4   

### 13.1.3 모델 저장소
MLOps에서 **모델 저장소(model registry)**는 머신러닝 모델을 체계적으로 관리하고 버전 제어하는 데 필수   
여러 머신러닝 파이프라인에서 다양한 실험을 통해 여러 버전의 모델이 생성되는데, 이를 활용하면 다양한 모델을 통합해서 관리할 수 있음   

그림 13.5   

### 13.1.4 모델 모니터링
머신러닝 모델은 학습 데이터를 통해 학습한 패턴을 바탕으로 예측을 수행하기 때문에 정상적으로 요청에 응답하고 있더라도 엉뚱한 값을 반환한 것은 아닌지 확인   

## 13.2 LLMOps는 무엇이 다를까?
### 13.2.1 상업용 모델과 오픈소스 모델 선택하기
LLMOps에서는 MLOps보다 훨씬 크고 다양한 일을 할 수 있는 모델을 다룬다는 점에서 큰 차이   

표 13.1   

### 13.2.2 모델 최적화 방법의 변화
LLMOps에서 다루는 LLM은 모델의 크기가 크기 때문에 일반적으로 사전 학습시키는 경우는 거의 없음   
오픈소스 모델을 선택했다면 미세 조정을 자유롭게 수행할 수 있지만, 상업용 모델을 선택했다면 미세 조정 기능을 지원하는 모델만 제한적으로 미세 조정할 수 있음   

표 13.2   

LLMOps에서 다루는 언어 모델은 모델의 크기가 크고 처음부터 학습시킬 때 들어가는 계산량이 크기 때문에 일반적으로 사전 학습하지 않고 사전 학습된 모델을 가져와 미세 조정하는 전이 학습을 기본으로 사용   
모델 개발 과정에서 학습할 때 설정한 하이퍼파라미터를 기록해 두고 이후 동일한 성능의 모델을 다시 만들 수 있도록 관리   

### 13.2.3 LLM 평가의 어려움
LLM은 다양한 작업이 가능하기 때문에 특정 작업의 성능 평가 방식으로 모두 평가할 수 없고 프롬프트에 따라 성능이 달라지기도 해서 명확한 기준을 잡기 어려움   

## 13.3 LLM 평가하기
### 13.3.1 정량적 지표
텍스트 생성 작업을 평가할 때 사용할 수 있는 대표적인 세 가지 정량 지표   
1. **BLEU(Bilingual Evaluation Understudy Score)** - 기계 번역 결과와 사람이 번역한 결과의 유사도를 측정하여 평가   
2. **ROUGE(Recall-Oriented Understudy for Gisting Evaluation)**- 모델이 생성한 요약문과 사람이 작성한 참조 요약문 사이의 n그램 중복도를 재현율 관점에서 측정   
3. **펄플렉시티(Perplexity)** - 모델이 새로운 단어를 생성할 때의 불확실성을 수치화한 것으로, 값이 낮을수록 모델의 예측 성능이 우수하다는 의미   

세 가지 정량 지표 모두 빠르게 언어 모델의 성능을 평가할 수 있다는 장점이 있지만 문장의 의미, 문법, 유창성 등 질적인 측면의 평가에는 한계가 있고 실제 사람의 주관적 판단과 불일치하는 경우가 많음   

### 13.3.2 벤치마크 데이터셋을 활용한 평가
**벤치마크 데이터셋** - 다양한 모델의 성능을 비교하기 위해 공통으로 사용하는 데이터셋, 대표적으로 (ARC, HellaSwag, MMLU 등)   

표 13.3   

W&B에서 새로운 한국어 LLM 리더보드인 호랑이(Horangi)가 공개   
문장의 생성 확률이 아니라 실제로 생성한 텍스트 결과가 A,B,C,D와 같이 정답과 일치하는지를 비교해 성능을 평가   

### 13.3.3 사람이 직접 평가하는 방식   
사람이 직접 평가하는 방식은 언어의 유창성과 같이 정량적인 지표로 평가하기 어려운 사항을 평가할 수 있다는 장점이 있지만 시간이 오래 걸리고 비용이 많이 든다는 단점이 있음   

### 13.3.4 LLM을 통한 평가
표 13.4   
첫 번째 턴에서 하나의 요청을 하고 응답 이후에 다시 두 번째 요청을 함   
어러 턴에 걸쳐 LLM이 사용자의 요구사항에 맞춰 대응하는지 확인하기 위해서임   

사람과 LLM의 평가가 80% 이상 일치했기 때문에 사람이 직접 평가할 때 드는 시간과 비용을 생각하면 LLM을 활용해 비교적 적은 비용으로 빠르고 정확하게 평가를 수행해서 사람이 직접 평가하는 양을 줄일 수 있음   

### 13.3.5 RAG 평가
그림 13.9   
1. **신뢰성(faithfulness)** - 생성된 응답이 검색된 맥락 데이터에 얼마나 사실적으로 부합하는 지 평가   
2. **답변 관련성(answer relevancy)** - 생성된 답변이 요청과 얼마나 관련성 있는지 평가
3. **맥락 관련성(context relevancy)** - 검색 결과인 맥락 데이터가 요청과 얼마나 관련 있는지 평가   

# 14 멀티 모달 LLM
## 14.1 멀티 모달 LLM이란
**멀티 모달 LLM** - 텍스트 뿐만 아니라 이미지, 비디오, 오디오, 3D 등 다양한 형식의 데이터를 이해하고 생성할 수 있는 LLM   
### 14.1.1 멀티 모달 LLM의 구성요소
멀티 모달 LLM은 일반적으로 다섯 가지 구성요소로 이뤄짐   
1. LLM은 뛰어난 이해 능력과 추론 능력을 갖고 있기 때문에 이미지 형식의 데이터를 **모달리티 인코더(modality encoder)**와 **입력 프로젝터(input projector)**를 통해 텍스트로 변환해 입력
2. LLM의 출력은 기본적으로 텍스트인데, **출력 프로젝터(output projector)**를 통해 이미지 형태의 데이터 출력이 필요한지 판단   
3. **모달리티 생성기(modality generator)**를 통해 특정 데이터 형식의 출력을 생성   

그림 14.2   

요소들의 오른쪽 아래에 있는 그림들은 학습 과정에서의 파라미터 업데이트 여부를 나타냄   

**모달리티 인코더** - 이미지, 비디오, 오디오 같이 텍스트 이외의 데이터 형식을 처리하기 위해 학습된 사전 학습 모델   
**비전 트랜스포머(vision transformer)** - 텍스트를 처리하기 위해 개발된 트랜스포머 아키텍처를 이미지에 적용한 모델   

그림 14.3   

이미지를 패치 단위로 자른 후 마치 텍스트에서 단어를 처리하는 것과 같이 일렬로 나열해 입력해 처리   
모달리티 인코더가 이미지 데이터를 처리해서 이미지 임베딩으로 변환했다면 입력 프로젝터는 이미지 임베딩을 LLM 백본이 이해할 수 있는 텍스트로 변환   
이미지를 생성하기 위해서는 크게 이미지 생성이 필요한지 판단하는 단계와 만약 필요하다면 어떤 이미지를 생성할지 정하는 단계로 나눌 수 있음   

### 14.1.2 멀티 모달 LLM 학습 과정
LLM과 마찬가지로 **사전 학습**과 지시 데이터셋을 활용한 **지시 학습(instruction tuning)**으로 나눔   
사전 학습 단계에서는 멀티 모달 데이터에 대한 전체적인 이해력을 습득하고 높이는 데 집중   
사전 학습이 끝난 후에는 멀티 모달 지시 튜닝 단계를 진행   
지시 - 모델이 이미지 캡션을 생성하거나 입력 이미지에 대한 질문 응답 같은 특정 멀티 모달 작업을 수행하도록 학습 시키는 것을 말함   

## 14.2 이미지와 텍스트를 연결하는 모델: CLIP
### 14.2.1 CLIP 모델이란
**CLIP(Contrastive Language_Image Pre-training)** - OpenAI에서 개발한 모델로 인터넷상에서 수집한 이미지와 캡션 데이터를 활용하여 이미지와 텍스트를 같은 벡터 공간에 임베딩하도록 만들어짐    

### 14.2.2 CLIP 모델의 학습 방법
CLIP은 이미지와 이미지에 대한 설명이 대응된 데이터인 이미지-텍스트 쌍을 활용   

그림 14.4   

유사한 데이터 쌍은 더 가까워지도록 하고 유사하지 않은 데이터 쌍은 더 멀어지도록 학습시키는 대조 학습을 통해 모델을 학습   

그림 14.5  

### 14.2.3 CLIP 모델의 활용과 뛰어난 성능
**제로샷 추론(Zero-shot prediction)** - 사전 학습 데이터 이외에 특정 작업을 위한 데이터로 미세 조정하지 않은 상태에서 추론을 수행하는 것   

그림 14.6   

CLRP 모델은 이미지와 텍스트 데이터 사이의 유사도 계산을 활용해 이미지 검색에도 활용할 수 있음   
이미지 검색 - 이미지와 텍스트의 유사도 기반으로 텍스트를 입력했을 때 유사한 이미지를 찾는 기능   

### 14.2.4 CLIP 모델 직접 활용하기
먼저 입력 데이터의 전처리를 담당하는 프로세서와 이미지와 텍스트 임베딩 모델을 가져옴   

```
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
probs
```

## 14.3 텍스트로 이미지를 생성하는 모델: DALL-E
### 14.3.1 디퓨전 모델 원리
**디퓨전 모델** - 물질이 농도가 높은 곳에서 낮은 곳으로 이동하는 현상인 확산 현상에서 영감을 받아 만들어진 생성 모델   

그림 14.10   

디퓨전 모델은 이미지에서 어떤 부분이 노이즈인지 예측하는 방식으로 학습하는데, 그 능력을 사용해 완전한 노이즈 상태의 이미지에서 노이즈를 예측하고 예측된 노이즈를 제거하면서 점차 완전한 노이즈에서 의미가 있는 이미지를 생성   

그림 14.11   

**인코더 디코더** - 입력 데이터의 차원을 낮추는 인코딩 단계와 차원을 높이는 디코딩 단계를 통해 데이터의 의미를 압축하기 위해 사용되는 모델 구조   
**U-Net** - 인코더-디코더 구조를 변형해 인코딩 단계의 고차원 정보를 디코딩 단계에도 활용함으로써 이미지의 위치 정보가 손실되는 것을 막을 수 있다는 장점이 있음   

그림 14.12   

사람이 원하는 이미지를 생성하기 위해서는 디퓨전 모델에 노이즈를 넣어주면서 원하는 결과물의 형태를 텍스트 임베딩으로 변환해 디퓨전 모델에 함께 입력으로 넣어줌   

그림 14.14   

### 14.3.2 DALL-E 모델
DALL-E 모델은 앞서 살펴본 CLIP 모델을 활용해 텍스트 임베딩을 만들고 텍스트 임베딩을 활용해 두 단계를 거쳐 이미지를 생성   

그림 14.15   

**프라이어(prior) 모델** - 텍스트 임베딩을 입력으로 받아 이미지 임베딩을 예측하는 디퓨전 모델   

그림 14.16   

**디코더** - 이미지 임베딩을 참조해 이미지를 생성하는 디퓨전 모델   
프라이어와 디코더 모두 디퓨전 모델 구조를 사용하지만 학습되는 데이터와 입력 및 출력의 차원이 다른 모델   

그림 14.17   

굳이 프라이어를 사용해야 할까?라는 의문이 들 수 있지만 결과적으로 프라이어와 디코더를 모두 사용했을 때 원하는 결과를 얻을 확률이 높아짐   

그림 14.18   

## 14.4 LLaVA
### 14.4.1 LLaVA의 학습 데이터
**LLaVA(Large Language and Visual Assistant) 모델** - 이미지를 인식하는 CLIP 모델과 LLM을 결합해 모델이 이미지를 인식하고 그 이미지에 대한 텍스트를 생성할 수 있는 모델   
데이터셋의 부족을 해결하기 위해서 GPT-4를 활용해 데이터셋을 생성   
텍스트만을 입력을 받기 때문에 이미지에 대한 설명과 위치 정보(Bounding Box)를 통해 이미지를 인식하도록 함   

그림 14.20   

GPT-4가 입력된 이미지 설명을 보고 다음 세 가지 유형의 텍스트를 생성하도록 함   

1. 대화: 사람이 이미지에 질문했을 때 어시스턴트(Assistant)가 이미지를 보고 답변하는 형식의 데이터   
2. 자세한 설명: GPT-4가 이미지 설명을 읽고 이미지에 대해 자세히 설명하도록 함   
3. 복합한 추론: 위의 두 유형이 단순히 이미지에 대한 인식과 설명이었다면, 답변을 위해 단계별 추론이 필요한 어려운 질문을 생성하고 답변하도록 함   

### 14.4.2 LLaVA 모델 구조
입력 이미지를 CLIP의 이미지 인코더를 통해 이미지 임베딩으로 만들고 간단한 선형 층을 통과해 LLM에 입력할 임베딩 토큰으로 만든 후 텍스트 지시사항은 토큰 임베딩으로 변환해 함께 입력으로 넣고 결과 생성   

그림 14.21   

### 14.4.3 LLaVA 1.5
한 층의 선형 층으로 이미지 임베딩을 토큰 임베딩으로 변환하던 구조를 2층의 MLP(Multi-Layer Perception)로 변경하는 간단한 수정만으로 성능을 대폭 끌어올림   

그림 14.22   

그림 14.23   

### 14.4.4 LLaVA NeXT
2024년 1월 30일에 발표된 LLaVA NeXT는 아래와 같은 사항들이 변경   

1. 기존 모델 대비 입력 이미지의 해상도가 4배 높아짐
2. 고품질의 지시 데이터셋을 구축해 시각적 추론 능력과 OCR 성능이 개선
3. 더 많은 시나리오에서 응답할 수 있어 다양한 애플리케이션에 활용
4. SGLangn 프레임워크를 사용해 추론 성능 향상   

표 14.1   

# 15 LLM 에이전트
## 15.1 에이전트란
### 15.1.1 에이전트의 구성요소
**AI 에이전트(agent)** - 주변 환경을 감각을 통해 인시갛고 의사결정을 내려 행동하는 인공적인 개체   
위와 같이 행동하기 위해서는 크게 세 가지 구성요소가 필요   

1. 감각(perception) - 이를 통해 외부 환경과 사용자의 요청을 인식
2. 두뇌(brain) - 이를 활용해 가지고 있는 지식이나 지금까지 기억을 확인해 보고 계획을 세우거나 추론을 통해 다음에 어떤 행동을 해야 할지 의사결정을 내림
3. 행동(action) - 문제를 해결하기 위해 취할 수 있는 적절한 도구를 선택해 행동   

그림 15.1   

### 15.1.2 에이전트의 두뇌
에이전트의 두뇌는 감각을 통해 현재 상황과 사용자의 요청을 인식한 것을 바탕으로 사용자의 요청이 무엇인지 현재 어떤 상황인지 이해하고 목표 달성을 위해 어떤 행동을 취할지 결정   
그 과정에서 지금까지 수행했던 사용자와의 대화나 행동을 저장한 기억(memory)을 확인하고, 상황을 이해하는 데 필요한 지식이 있다면 검색해서 활용   
이를 바탕으로 에이전트는 목표 달성을 위한 작업을 세분화하는 계획 세우기(planning) 단계를 거치고 바로 다음에 어떤 행동이 필요한지 결정해 단계로 넘어감   

그림 15.2   

### 15.1.3 에이전트의 감각
비디오와 텍스트 멀티 모달 모델인 비디오 라마(Video-LLaMA) 모델은 비디오에 있는 영상 정보와 음성 정보를 각각 비디오 인코더와 오디오 인코더를 통해 처리   

그림 15.3   

### 15.1.4 에이전트의 행동
LLM은 텍스트만을 생성할 수 있기 때문에 외부에 영향을 미치는 행동을 하기 위해서는 LLM이 사용할 수 있는 도구를 제공해야 함   

그림 15.4   

## 15.2 에이전트 시스템의 형태
### 15.2.1 단일 에이전트
목표만 입력하면 알아서 작업을 수행하는 AutoGPT의 경우 입력받은 프롬프트를 통해 모든 결정을 내림   
단일 에이전트의 경우 모든 과정을 스스로 처리하기 때문에 매우 편리하고 범용적인 작업을 처리할 수 있는 프롬프트로 동작하기 때문에 다양한 작업에서 사용될 수 있지만 작업을 수행하는 과정에서 길을 잃을 가능성도 큼   

### 15.2.2 사용자와 에이전트의 상호작용
에이전트를 사용하면 에이전트가 도구를 활용해 직접 외부 정보를 검색하거나 파일로 정리하는 등 행동을 수행할 수 있어 사용자의 개입이 덜 필요   
그렇지만 동등한 지위에서 상호작용하거나 감정적인 교감이 필요한 경우에느 사용자의 개입이 불가피한 경우도 있음   

그림 15.6   

### 15.2.3 멀티 에이전트
**멀티 에이전트** - 단일 에이전트와 달리 각 에이전트마다 서로 다른 프로필(profile)을 주고 작업을 수행하여 관련된 작업의 전문성을 높에 결과 품질을 높일 수 있음    
맞춤형 에이전트를 여러 개 만들어 작업을 수행할 때 문제 해결 확률을 더 높일 수 있고 멀티 에이전트끼리는 대화를 통해 작업의 진행 상황과 작업 결과를 공유   
수평형 대화(joint chat) - 여러 에이전트가 대화를 나누며 문제를 해결할 때는 모두 함께 대화를 진행   
위계형 대화(hierarchical chat) - 대화를 주도하는 매니저나 상황에 따라 작업할 에이전트를 선택하고 대화가 진행   

그림 15.7   

## 15.3 에이전트 평가하기
에이전트 평가 방식은 크게 사람이 평가하는 주관적인 방식과 테스트 데이터로 평가하는 객관적인 방식으로 나눌 수 있음   
**튜링 테스트(turing test)** - 에이전트와 사람의 결과물을 구분할 수 있는지 확인하는 평가 방식   

그림 15.9   

에이전트를 평가하는 큰 기준 네 가지   
1. 유용성 - 작업을 자동으로 수행하기 위해 에이전트를 사용하는 경우 작업 성공률 같은 기준으로 유용성을 평가할 수 있음   
2. 사회성 - 언어를 얼마나 숙련되게 사용하는지, 작업을 진행하는 과정에서 협력이나 협상이 필요한 경우 얼마나 뛰어난지, 역할을 부여한 경우 그 역할에 얼마나 부합하게 행동하는지 등을 평가   
3. 가치관 - 거짓말을 지어내지 않고 신뢰할 수 있는 정확한 정보를 전달하는지, 차별적이거나 편향이 있는지, 사회에 해가 될 수 있는 정보를 전달하는지 평가   
4. 진화 능력 - 지속적인 학습, 스스로 목표를 설정하고 달성하는 학습 능력, 환경에 적응하는 능력   

칭화대에서 8개의 작업에서 LLM 에이전트를 평가할 수 있는 평가용 데이터셋인 AgentBench를 구축   

그림 15.10   

상용 LLM이 오픈소스 모델에 비해 훨씬 더 뛰어난 에이전트 성능을 보여줌   
에이전트가 실패하는 대표적인 경우로 대화가 여러 턴에 걸쳐 진행되면서 추론 및 의사결정 능력이 떨어지는 경우와 요청한 출력 형식으로 응답하지 않은 경우로 꼽음   

그림 15.11   

## 15.4 실습: 에이전트 구현
### 15.4.1 AutoGen 기본 사용법
마이크로소프트가 공개한 에이전트 프레임워크인 AutoGen을 사용해 에이전트를 구현   
AutoGen에는 크게 두 종류의 에이전트가 있음   
1. UserProxyAgent - 사용자의 역할을 대신함   
2. AssistantAgent - 사용자의 요청을 처리함   
```
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy",
  is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
  human_input_mode="NEVER",
  code_execution_config={"work_dir": "coding", "use_docker": False})
```

그림 15.12   

### 15.4.2 RAG 에이전트
```
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "collection_name": "default-sentence-transformers"
    },
)

assistant.reset()
ragproxyagent.initiate_chat(assistant, problem="AutoGen이 뭐야?")

# assistant (to ragproxyagent):
# AutoGen은 여러 에이전트가 상호 대화하여 작업을 해결할 수 있는 LLM(Large Language Model) 애플리케이션 개발을 가능하게 하는 프레임워크입니다. 
```
AutoGen 에이전트는 사용자 정의 가능하며, 대화 가능하고, 인간 참여를 원활하게 허용합니다. LLM, 인간 입력, 도구의 조합을 사용하는 다양한 모드에서 작동할 수 있습니다.

AutoGen에서는 RAG를 구현할 때 기본적으로 텍스트 임베딩 모델에 Sentence-Transformers 라이브러리를 사용하고, 벡터 데이터베이스로는 크로마를 사용   
총 4개의 에이전트가 협업하는 그룹챗(GroupChat)을 생성   

```
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# RAG를 사용하지 않는 사용자 역할 에이전트
user = autogen.UserProxyAgent(
    name="Admin",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="The boss who ask questions and give tasks.",
    code_execution_config=False,
    default_auto_reply="Reply `TERMINATE` if the task is done.",
)
# RAG를 사용하는 사용자 역할 에이전트
user_rag = RetrieveUserProxyAgent(
    name="Admin_RAG",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    code_execution_config=False,
    retrieve_config={
        "task": "code",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/samples/apps/autogen-studio/README.md",
        "chunk_token_size": 1000,
        "collection_name": "groupchat-rag",
    }
)
# 프로그래머 역할의 에이전트
coder = AssistantAgent(
    name="Senior_Python_Engineer",
    is_termination_msg=termination_msg,
    system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)
# 프로덕트 매니저 역할의 에이전트
pm = autogen.AssistantAgent(
    name="Product_Manager",
    is_termination_msg=termination_msg,
    system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
    llm_config=llm_config,
)

PROBLEM = "AutoGen Studio는 무엇이고 AutoGen Studio로 어떤 제품을 만들 수 있을까?"
``` 

### 15.4.3 멀티 모달 에이전트

```
def dalle_call(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1) -> str:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    img_data = get_image_data(image_url)
    return img_data

class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)

        try:
            config_list = llm_config["config_list"]
            api_key = config_list[0]["api_key"]
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)

    def generate_dalle_reply(self, messages, sender, config):
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]
        img_data = dalle_call(client=self.client, prompt=prompt)
        plt.imshow(_to_pil(img_data))
        plt.axis("off")
        plt.show()
        return True, 'result.jpg'
```

이미지를 설명하는 에이전트와 그림을 생성하는 에이전트가 협력해 이미지를 입력했을 때 유사한 이미지를 알아서 생성   

# 16 새로운 아키텍처
2023년 12월 **맘바(Mamba)** 아키텍처가 "트랜스포머와 성능이 비슷하거나 뛰어나면서 추론 속도가 5배"라고 주장   
맘바는 RNN을 개선한 모델이라고 할 수 있는데, SSM은 그중에서 속도를 높이기 위한 전략이고 선택 메커니즘은 문장의 맥락을 효율적으로 압축해 성능을 높이려는 전략   

그림 16.1   

## 16.1 기존 아키텍처의 장단점
표 16.1   

## 16.2 SSM
**SSM(State Space Model)** - 내부 상태를 가지고 시간에 따라 달라지는 시스템을 해석하기 위해 사용하는 모델링 방법   
아래 식에서 h는 모델 내부의 상태이고, x는 모델에 들어오는 압력, A,B,C,D는 입력, 상태와 출력 사이의 관계를 연결하는 행렬   
```
h(t) = Ah(t-1) + Bx(t)
y(t) = Ch(t) + Dx(t)
```

그림 16.2   

입력, 상태, 출력 사이의 관계를 표현하는 A,B,C,D 모두 행렬이기 때문에 선형(linear) 관계를 가정   

### 16.2.1 S4
**S4(Structed State Space for Sequence Modeling)** - 대표적인 SSM 모델 중 하나로 계산 효율성을 높여 쉽게 모델을 학습시킬 수 있도록 함   
학습을 효율적으로 만들기 위해 A,B,C,D가 시간에 따라 변하지 않도록 고정 - **선형 시간 불변성(linear time invariant)**   
```
h(0) = Bx(0)
h(1) = Ah(0) + Bx(1) = ABx(0) + Bx(1)
h(2) = Ah(1) + Bx(2) = A^2Bx(0) + ABx(1) + Bx(2)
...
```
A 행렬이 반복적으로 곱해지게 되는데, 이때 더 효율적인 연산을 위해 S4에서는 A 행렬을 **대각 행렬(diagonal matrix)**로 만듦   
학습에서는 컨볼루션(convolution)연산을 통해 병렬로 계산해 학습 속도를 높이고 추론에서는 순차적인 방식으로 계산하는 CNN과 RNN을 결합한 방식을 사용   

그림 16.3   

그러나 S4는 언어 모델링처럼 이산적이고(discrete) 정보 집약적인 작업에선느 트랜스포머보다 낮은 성능을 보임   

## 16.3 선택 메커니즘
**LSTM(Long Short Term Memory)** - RNN에서 발전시킨 이 모델은 맥락을 더 효율적으로 압축하기 위해 입력을 얼마나 반영할지, 기존 상태를 얼마나 망각할지 결정하는 게이트를 추가   

그림 16.4   

계산 속도를 위해 필요했던 선형 시간 불변성 제약을 성능을 위해 포기하면서 맘바는 계산 효율을 높여야 하는 도전에 직면   

1. 커널 퓨전(kernel fusion) - GPU IO 줄이기
2. 중간 결과물 재계산 - 역전파에 필요한 중간 결과물을 저장하지 않고 필요할 때 재계산
3. 병렬 스캔(parallel scan) 사용   

GPU IO를 최소화 하기 위해 중간 과정을 저장하지 않는데, 이렇게 하면 연산량이 늘어나지만 IO가 대폭 줄어 결과적으로는 연산 시간이 단축   
병렬 스캔은 입력에 따라 가중치가 달라지기 때문에 기본적으로 재귀적인 성질을 갖는 맘바에 병렬 연산을 가능하게 함   

## 16.4 맘바
### 16.4.1 맘바의 성능
표 16.3   
맘바 아키텍처는 높은 성능과 빠른 학습, 추론 속도를 갖춰 연산 부담이 큰 트랜스포머 아키텍처의 강력한 대안으로 주목   
그러나 이미 많은 언어 모델이 트랜스포머를 기반으로 동작하고 있기 때문에 완전히 대체하기는 어려움   

### 16.4.2 기존 아키텍처와의 비교
표 16.4   

## 16.5 코드로 보는 맘바
맘바 모델의 핵심은 동일한 맘바 블록을 반복적으로 통과하는 구조   

그림 16.9   

맘바 코드의 중요한 인자   
1. d_model - 토큰 임베딩 차원
2. d_inner - 모델 내부적으로 토큰 임베딩을 확장해 사용하는 차원   
3. d_state - 선택 메커니즘에서 입력을 확장할 때 사용하는 상태 차원   

```
class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        # ssm 내부에서 사용
        # 입력 x를 확장해 Δ, B, C를 위한 벡터를 생성하는 층
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        # dt_rank차원을 d_inner차원으로 확장해 Δ 생성하는 층
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'd_state -> d_model d_state',
        d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    def forward(self, x):
        (b, l, d_model) = x.shape
        x_and_res = self.in_proj(x) # shape (b, l, 2 * d_inner)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner],
        dim=-1)
        x = rearrange(x, 'b l d_inner -> b d_inner l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_inner l -> b l d_inner')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
    return output
```