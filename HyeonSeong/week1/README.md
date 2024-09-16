# LLM을 활용한 실전 AI 애플리케이션 개발
------------------
# 7 모델 가볍게 만들기

## 7.1 언어 모델 추론 이해하기

### 7.1.1 언어 모델이 언어를 생성하는 방법
언어 모델이 텍스트 생성을 마치는 이유는 크게 두가지
- 다음 토큰으로 생성 종료를 의미하는 특수 토큰을 생성하는 경우
- 사용자가 최대 길이로 설정한 길이에 도달하면 더 이상 생성하지 않고 종료

그림 7.1

언어모델은 입력 텍스트를 기반으로 바로 다음 토큰만 예측하는 자기 회귀적(auto-regressive)인 특성을 보유
동일한 토큰이 반복해서 입력으로 들어가면 동일한 연산을 반복적으로 수행하기 때문에 비효율적

그림 7.3

### 7.1.2 중복 연산을 줄이는 KV 캐시
**KV 캐시** - 셀프 어텐션 연산 과정에서 동일한 입력 토큰에 대해 중복 계산이 발생하는 비효율을 줄이기 위해 먼저 계산했던 키와 값 결과를 메모리에 저장해 활용하는 방법   

그림 7.4

KV 캐시 메모리의 계산식

**KV 캐시 메모리 = 2바이트 X 2(키의 값) X (레이어 수) X (토큰 임베딩 차원) X (최대 시퀀스 길이) X (배치 크기)**

처음 2는 fp16의 데이터 형식이기 때문에 **2바이트**    
두번째 2는 키 캐시와 값 2개를 저장하기 때문에 **2**    
어텐션의 연산 결과는 레이어 수 만큼 생기기 때문에 **레이어 수**   
토큰 임베딩을 표현하는 차원의 수만큼 수를 저장하기 때문에 **차원 수**   
최대로 생성하려는 시퀀스 길이만큼의 메모리를 미리 확보하기 때문에 **시퀀스 길이**   
배치 크기에 따라 저장하는 데이터가 달라지므로 **배치 크기**   

### 7.1.3 GPU 구조와 최적의 배치 크기
서빙이 효율적인지 판단하는 큰 기준 3 가지
1. 비용
2. 처리량(throughput) - 시간당 처리한 요청(쿼리) 수
3. 지연 시간(latency) - 하나의 토큰을 생성하는 데 걸리는 시간   

그림 7.6   
GPU는 여러 스트리밍 멀티프로세서(streaming multiprocessors)로 구성   
각각의 SM에는 연산을 수행하는 부분과 값을 저장하는 SRAM(Static Random Access Memory)가 존재   
연산을 수행하는 부분과 가까운 SRAM에는 큰 메모리를 갖기 어렵기 때문에 큰 고대역폭 메모리(High Bandwidth Memory)에 큰 데이터를 저장    
추론을 수행할 때는 연산을 수행하는 시간과 메모리를 이동시키는 시간이 걸림   
모델의 이동 과정과 연산 수행 과정은 함께 진행되기 때문에 두 가지 시간이 같을 때가 최적의 배치 크기   
- 메모리 바운드: 배치 크기가 작아서 모델 파라미터를 이동시키느라 연산이 멈추는 비효율 발생
- 연산 바운드: 배치 크기가 커서 연산에 더 오랜 시간이 발생   

그림 7.9

### 7.1.4 KV 캐시 메모리 줄이기
**멀티 쿼리 어텐션**(multi-query attention) - 모든 쿼리 벡터가 하나의 키와 값 벡터를 공유하는 방식   
**그룹 쿼리 어텐션**(grouped-query attention) - 멀티 쿼리 어텐션의 성능 감소로 인해 그보다는 키와 값의 수를 늘린 방식   

그림 7.11

키의 값의 수를 줄이면서 크게 추론 속도 향상과 KV 캐시 메모리의 감소에 대한 효과가 있음   
멀티 쿼리 어텐션의 경우 멀티 헤드 어텐션과 비교했을 때 성능 저하가 뚜렷하기 때문에 키와 값을 줄인 이후에 기존의 학습 데이터로 추가 학습을 수행   
그에 반해 그룹 쿼리 어텐션은 멀티 헤드 어텐션과 성능 차이가 적음   

## 7.2 양자화로 모델 용량 줄이기
양자화를 수행하는 시점에 따라 학습 후 양자화 / 양자화 학습으로 나눔   
LLM의 경우 학습에 많은 자원이 들기 때문에 새로운 학습이 필요한 양자화 학습보다는 학습 후 양자화를 주로 활용   
### 7.2.1 비츠앤바이츠
**비츠앤바이츠** - 8비트 연산을 수행하면서도 성능 저하가 거의 없는 8비트 행렬 연산과 4비트 정규 분포 양자화 방식을 제공하는 양자화 라이브러리   
8비트 행렬 연산 - 입력값 중 크기가 큰 이상치가 포함된 열은 별도로 분리해서 16비트 그대로 계산

그림 7.15

### 7.2.2 GPTQ
**GPTQ**(GPT Quantization) - 모델에 입력 X를 넣었을 때와 양자화 이후의 모델에 입력 X를 넣었을 때 오차가 가장 작아지도록 모델의 양자화를 수행   

그림 7.16

흰색 열의 양자화를 수행하고 양자화를 위해 준비한 데이터를 입력한 결과가 이전과 최대한 가까워지도록 아직 양자화하지 않은 오른쪽 부분의 파라미터를 업데이트   

### 7.2.3 AWQ
**AWQ**(Activation-aware Weight Quantization) - 모든 파라미터가 동등하게 중요하지는 않으며 특별히 중요한 파라미터의 정보를 유지하면 양자화를 수행하면서도 성능 저하를 막을 수 있다는 아이디어에서 출발   
어떤 파라미터가 중요한 지 크게 두가지로 판단
1. 모델 파라미터의 값이 크다
2. 입력 데이터 활성화 값이 큰 채널

위의 기준을 통해 상위 1%에 해당하는 모델 파라미터를 찾고 해당 파라미터는 기존 모델의 데이터 타입인 FP16을 유지하고 나머지는 양자화 진행하니 성능 저하가 거의 발생하지 않음   

그림 7.20

중요한 파라미터에만 1보다 큰 스케일러 값을 곱하는 방식으로 양자화 진행   
그러나 스케일러 S가 2일 때까지는 성능이 향상되지만 2를 넘어가는 경우 성능이 다시 하락하는 사실 확인   

그림 7.21

스케일러가 큰 경우 가장 큰 수의 값이 변화하기 때문에 다른 파라미터에 영향을 주어 정보 소실이 발생   

## 7.3 지식 증류 활용하기
**지식 증류**(knowledge distillation) - 더 크고 성능이 높은 선생 모델(teacher model)의 생성 결과를 활용해 더 작고 성능이 낮은 학생 모델(student model)을 만드는 방법   
그림 7.22

제퍼-7B-베타 모델의 경우 개발의 지시 데이터셋의 구축과 선호 데이터셋의 구축에 모두 LLM을 사용   
그림 7.23   
사람의 리소스가 필요한 작업에 모델을 활용해 개발 속도를 높이고 자원을 아낄 수 있음   

# 8 sLLM 서빙하기
## 8.1 효율적인 배치 전략
### 8.1.1 일반 배치(정적 배치)
**일반/정적 배치**(naive/static batching) - 가장 기본적인 방식으로 한 번에 N개의 입력을 받아 모두 추론이 끝날 때까지 기다리는 방식   
단점
- 다른 데이터의 추론을 기다리느라 결과를 반환하지 못하고 대기하는 비효율 발생   
- 생성이 일찍 종료되는 문장이 있으면 결과적으로 배치 크기가 작아지는 리소스 낭비

### 8.1.2 동적 배치
**동적 배치**(dynamic batching) - 비슷한 시간대에 들어오는 요청을 하나의 배치로 묶어 배치 크기를 키우는 전략   

그림 8.2

그러나 생성하는 토큰 길이 차이로 인해 처리하는 배치 크기가 점차 줄어들어 GPU를 비효율적으로 사용   

### 8.1.3 연속 배치
**연속 배치**(continuous batching) - 일반 배치와 달리 한 번에 들어온 배치 데이터의 추론이 모두 끝날 때까지 기다리지 않고 하나의 토큰 생성이 끝날 때마다 생성이 종료된 문장은 제거하고 새로운 문장을 추가   

그림 8.3

사전 연산과 디코딩은 처리 방식이 다르기 때문에 처리 중인 문장과 대기 중인 문장의 비율을 지켜보고 특정 조건을 달성했을 때 추가   

## 8.2 효율적인 트랜스포머 연산
기존의 절대적 위치 인코딩의 경우 학습 데이터보다 긴 입력 데이터가 들어올 때 성능이 크게 저하되는 단점이 존재
### 8.2.1 플래시어텐션
**플래시어텐션**(FlashAttention) - 트랜스포머가 더 긴 시퀀스를 처리하도록 만들기 위해 개발   
마스크, 소프트맥스, 드롭아웃과 같이 큰 메모리를 사용하는 연산이 오래 걸리는 이유는 GPU에서 메모리를 읽고 쓰는 데 오랜 시간이 걸리기 때문   
SRAM이 빠르지만 메모리의 크기가 작기 때문에 대부분의 읽기 쓰기 작업은 **HBM**에서 진행   
블록 단위로 어텐션 연산을 수행하고 전체 어텐션 행렬을 쓰거나 읽기 않는 방식을 도입   
이를 활용하면 HBM이 아닌 SRAM에 데이터를 읽고 쓰면서 더 빠르게 연산 수행 가능   

그림 8.7

연산량이 증가하지만 메모리를 읽고 쓰는 양이 크게 줄어들면서 실행시간은 오히려 1/5정도로 감소   

표 8.1

### 8.2.2 플래시어텐션 2
**플래시어텐션2**는 플래시어텐션에 비해 크게 두 가지를 개선해 2배 정도 속도를 향상함
1. 행렬 곱셈이 아닌 연산 줄이기
2. 시퀀스 길이 방향의 병렬화 추가

그림 8.14

GPU를 효율적으로 활용하기 위해서는 충분한 수의 스레드 블록이 있어야 하는데 아래 그림과 같이 스퀀스 길이 방향으로 여러 개의 묶음으로 나눠 사용하는 스레드 블록 수를 늘리는 식으로 작동   

그림 8.15

### 8.2.3 상대적 위치 인코딩
최초의 트랜스포머 아키텍처에서는 토큰의 위치에 따라 사인과 코사인 수식으로 정해진 값을 더해줌   
이렇게 사인과 코사인을 사용해 위치 인코딩을 더하는 방식을 **사인파(sinusoidal)위치 인코딩**이라고 부름   
그러나 학습 데이터보다 더 긴 입력이 들어오면 언어 모델의 생성 품질이 빠르게 떨어진다는 한계가 있음   
이를 극복하기 위해 토큰과 토큰 사이의 상대적인 위치 정보를 추가하는 **상대적 위치 인코딩**(relative positional encoding) 방식이 활발히 연구됨   
**RoPE**(Rotary Positional Encoding) - 각각의 토큰 임베딩을 토큰 위치에 따라 회전시키는 방식   
토큰 사이의 위치 정보가 두 임베딩 사이의 각도를 통해 모델에 반영

그림 8.18

**ALiBi**(Attention with Linear Biases) - 쿼리와 키 벡터를 곱한 어텐션 행렬에 오른쪽에서 왼쪽으로 갈수록 더 작은 값을 더하는 방식   
간단한 인코딩 방식을 사용하기 떄문에 학습과 추론에도 별도로 처리 시간을 추가하지 않음   

그림 8.21

## 8.3 효율적인 추론 전략
### 8.3.1 커널 퓨전
GPU에서 연산은 커널 단위로 이루어지는데 커널마다 고대역폭 메모리에서 데이터를 읽어오고 연산 결과를 쓰는 오버헤드가 발생   
**커널 퓨전**(kernel fusion) - 연산을 하나로 묶어 오버헤드를 줄이는 방식   

그림 8.22

### 8.3.2 페이지어텐션
기존의 KV 캐시는 앞으로 사용할 수도 있는 메모리를 미리 잡아두면서 GPU 메모리를 많이 낭비한다는 문제 발생   

그림 8.25

'연속적인 물리 메모리'를 사용하기 위해 미리 메모리를 준비하기 때문에 발생   
**페이지어텐션** - 운영체제의 가상 메모리 개념을 빌려와 중간에서 논리적 메모리와 물리적 메모리를 연결하는 블록 테이블을 관리   

그림 8.26

다양한 디코딩 방식에서 동일한 입력의 프롬프트에 대한 메모리를 공유하는 방식인 **병렬 샘플링**(parallel sampling)을 활용해서 메모리를 절약   
다른 토큰을 생성하기 때문에 이를 분리하기 위해서 **참조 카운트**(reference count)라는 개념을 활용   
**참조 카운트** - 물리적 블록을 공유하고 있는 논리적 블록 수를 의미   

### 8.3.3 추측 디코딩
**추측 디코딩**(speculative decoding) - 쉬운 단어는 더 작고 효율적인 모델이 예측하고 어려운 단어는 더 크고 성능이 좋은 모델이 예측하는 방식   
작은 드래프트 모델(draft model)과 큰 타깃 모델(target model)이라는 2개의 모델을 활용   
먼저 드래프트 모델이 토큰을 생성하고 타깃 모델은 생성했을 결과와 동일한지 계산해 승인하거나 비승인하는 방식

그림 8.28

2개의 모델을 사용하기 때문에 시스템 복잡도가 올라가지만 이를 해결하기 위해 하나의 원본 모델 내에서 여러 토큰을 예측하는 메두사(Medusa)와 같은 방식도 존재   

## 8.4 실습: LLM 서빙 프레임워크
- 오프라인 추론: 대량의 입력 데이터에 대해 추론을 수행해 충분히 큰 배치 크기를 활용할 수 있는 추론
- 온라인 추론: 사용자의 요청에 따라 모델 추론을 수행하는 방식   
### 8.4.1 오프라인 서빙
```
import time

for max_num_seqs in [1, 2, 4, 8, 16, 32]:
  start_time = time.time()
  llm.llm_engine.scheduler_config.max_num_seqs = max_num_seqs
  sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=128)
  outputs = llm.generate(dataset['prompt'].tolist(), sampling_params)
  print(f'{max_num_seqs}: {time.time() - start_time}')
```

표 8.3
### 8.4.2 온라인 서빙
```
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="shangrilar/yi-ko-6b-text2sql",
                                 prompt=dataset.loc[0, 'prompt'], max_tokens=128)
print("생성 결과:", completion.choices[0].text)
```

# 9 LLM 애플리케이션 개발하기
## 9.1 검색 증강 생성(RAG)
LLM의 답변은 근거나 출처가 불명확하고 부정확한 정보를 지어내는 환각 현상이 존재   
**RAG**(Retrieval Augmented Generation) - 질문이나 요청만 전달하고 생성하는 것이 아니라 답변에 필요한 충분한 정보와 맥락을 제공하고 답변하도록 하는 방법   

그림 9.2

**LLM 오케스트레이션 도구** - 인터페이스, 임베딩 모델, 벡터 데이터베이스 등 LLM 애플리케이션을 위한 다양한 구성요소를 연결하는 프레임워크   
### 9.1.1 데이터 저장
**데이터 소스** - 텍스트, 이미지와 같은 비정형 데이터가 저장된 데이터 저장소   
**임베딩 모델** - 비정형 데이터를 입력했을 때 그 의미를 담은 임베딩 벡터로 변환하는 모델   
**벡터 데이터베이스** - 임베딩 벡터의 저장소이고 입력한 벡터와 유사한 벡터를 찾는 기능 제공   

그림 9.5

### 9.1.2 프롬프트에 검색 결과 통합
LLM은 결과를 생성할 때 프롬프트만 입력으로 받기 때문에 사용자의 요청과 관련이 큰 문서를 벡터 데이터베이스에서 찾고 검색 결과를 프롬프트에 통합해야 함   
결과의 정확도를 위해 매번 질문에 관련된 정보를 수동으로 찾아 입력해 줄 수는 없기 때문에 프로그래밍 방식으로 관련된 정보를 찾아 프롬프트에 넣을 수 있어야 함   

그림 9.9

### 9.1.3 실습: 라마인덱스로 RAG 구현하기
대표적인 LLM 오케스트레이션 라이브러리인 **라마인덱스**를 사용해 진행   
```
query_engine = index.as_query_engine(similarity_top_k=1)
response = query_engine.query(
    dataset[0]['question']
)
print(response)
# 장마전선에서 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 한 달 정도입니다.
```
다양한 구성요소와 과정이 필요하지만, 라마인덱스를 사용하면 단 두 줄의 코드만으로 유사한 텍스트를 검색하고 생성하는 과정을 전부 수행 가능   
```
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
## 9.2 LLM 캐시
LLM 캐시는 추론을 수행할 때 사용자의 요청과 생성 결과를 기록하고 이후에 동일하거나 비슷한 요청이 들어오면 새롭게 텍스트를 생성하지 않고 이전의 생성 결과를 가져와 바로 응답   

### 9.2.1 LLM 캐시 작동 원리
LLM 캐시는 프롬프트 통합과 LLM 생성 사이에 위치하여 캐시 요청을 통해 이전에 동일하거나 유사한 요청이 있었는지 확인   
크게 두 가지 방식으로 나눌 수 있음   

1. **일치 캐시**(exact cash) - 요청이 완전히 일치하는 경우 저장된 응답을 반환
2. **유사 검색 캐시**(similar search) - 문자열 그대로가 아닌 문자열의 임베딩 벡터를 비교하여 유사한 요청이 있었는지 확인   

### 9.2.2 실습: OpenAI API 캐시 구현
파이썬 딕셔너리와 오픈소스 벡터 데이터베이스 크로마(Chroma)를 사용해 기능 구현   
```
class OpenAICache:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.cache = {}

    def generate(self, prompt):
        if prompt not in self.cache:
            response = self.openai_client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
            )
            self.cache[prompt] = response_text(response)
        return self.cache[prompt]

openai_cache = OpenAICache(openai_client)

question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
for _ in range(2):
    start_time = time.time()
    response = openai_cache.generate(question)
    print(f'질문: {question}')
    print("소요 시간: {:.2f}s".format(time.time() - start_time))
    print(f'답변: {response}\n')

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 2.74s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다. 이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.

# 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
# 소요 시간: 0.00s
# 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다. 이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.
```
입력받은 prompt가 self.cache에 없다면 새롭게 저장하고 동일한 프롬프트가 있다면 저장된 응답을 반환   

## 9.3 데이터 검증
### 9.3.1 데이터 검증 방식
**데이터 검증** - 벡터 검색 결과나 LLM 생성 결과에 포함되지 않아야 하는 데이터를 필터링하고 답변을 피해야 하는 요청을 선별함으로써 새롭게 생성된 텍스트로 인해 생길 수 있는 문제를 줄이는 방법   
일종의 가이드라인으로 사용할 수 있는 방법은 크게 네 가지

1. 규칙 기반 - 문자열 매칭이나 정규 표현식을 활용해 데이터를 확인   
2. 분류 또는 회귀 모델 - 명확한 문자열 패턴이 없는 경우 별도의 모델(긍부정 분류 모델)을 만들어 활용   
3. 임베딩 유사도 기반 - 민감한 임베딩 벡터를 만들어 이와 관련된 답변을 피하도록 만들 수 있음   
4. LLM 활용 - LLM을 활용해 응답이 적절하지 않은 경우 이를 다시 생성하거나 삭제   

### 9.3.2 데이터 검증 실습
엔비디아에서 개발한 NeMo-Guardrails 라이브러리를 활용해 특정 주제에 대한 답변을 피하는 기능 구현   
```
colang_content_cooking = """
define user ask about cooking
    "How can I cook pasta?"
    "How much do I have to boil pasta?"
    "파스타 만드는 법을 알려줘."
    "요리하는 방법을 알려줘."

define bot refuse to respond about cooking
    "죄송합니다. 저는 요리에 대한 정보는 답변할 수 없습니다. 다른 질문을 해주세요."

define flow cooking
    user ask about cooking
    bot refuse to respond about cooking
"""
# initialize rails config
config = RailsConfig.from_content(
    colang_content=colang_content_cooking,
    yaml_content=yaml_content
)
# create rails
rails_cooking = LLMRails(config)

rails_cooking.generate(messages=[{"role": "user", "content": "사과 파이는 어떻게 만들어?"}])
# {'role': 'assistant',
#  'content': '죄송합니다. 저는 요리에 대한 정보는 답변할 수 없습니다. 다른 질문을 해주세요.'}
```
## 9.4 데이터 로깅
**데이터 로깅** - 사용자의 입력과 LLM이 생성한 출력을 기록   
LLM의 경우 입력이 동일해도 출력이 달라질 수 있기 때문에 어떤 입력에서 어떤 출력을 반환했는지 반드시 기록해야 함   
대표적인 로깅 도구 중 하나인 W&B(Weight adn Bias)에서 제공하는 Trace 기능을 활용하면 요청과 응답을 기록할 수 있음   

### 9.4.1 OpenAI API 로깅
```
import datetime
from openai import OpenAI
from wandb.sdk.data_types.trace_tree import Trace

client = OpenAI()
system_message = "You are a helpful assistant."
query = "대한민국의 수도는 어디야?"
temperature = 0.2
model_name = "gpt-3.5-turbo"

response = client.chat.completions.create(model=model_name,
                                        messages=[{"role": "system", "content": system_message},{"role": "user", "content": query}],
                                        temperature=temperature
                                        )

root_span = Trace(
      name="root_span",
      kind="llm",
      status_code="success",
      status_message=None,
      metadata={"temperature": temperature,
                "token_usage": dict(response.usage),
                "model_name": model_name},
      inputs={"system_prompt": system_message, "query": query},
      outputs={"response": response.choices[0].message.content},
      )

root_span.log(name="openai_trace")
```
### 9.4.2 라마인덱스 로깅
```
from datasets import load_dataset
import llama_index
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import set_global_handler
# 로깅을 위한 설정 추가
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
set_global_handler("wandb", run_args={"project": "llamaindex"})
wandb_callback = llama_index.core.global_handler
service_context = ServiceContext.from_defaults(llm=llm)

dataset = load_dataset('klue', 'mrc', split='train')
text_list = dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

print(dataset[0]['question']) # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?

query_engine = index.as_query_engine(similarity_top_k=1, verbose=True)
response = query_engine.query(
    dataset[0]['question']
)
```
# 10 임베딩 모델로 데이터 의미 압축하기
## 10.1 텍스트 임베딩 이해하기
**텍스트 임베딩 / 문장 임베딩** - 여러 문장의 텍스트를 임베딩 벡터로 변환하는 방식   
### 10.1.1 문장 임베딩 방식의 장점
문장 임베딩 방식을 사용하면 서로 다른 텍스트를 마치 사람이 이해하는 것처럼 서로 유사한지, 관련이 있는지 판단할 수 있다는 장점이 있음   

### 10.1.2 원핫 인코딩
데이터를 그대로 숫자로 변환한다고 하면 숫자를 비교하게 되어 오해가 생길 수 있기 때문에 **원핫 인코딩**(one-hot encoding)이 표현되었음   
**원핫 인코딩** - 숫자를 0,1의 행렬로 표시하여 데이터 사이의 의도하지 않은 관계가 담기는 것을 방지   
그러나 충분히 관련이 있는 단어 사이의 관계도 표현할 수 없다는 치명적인 단점이 있음   

### 10.1.3 백오브워즈
**백오브워즈**(Bag of Words) - '비슷한 단어가 많이 나오면 비슷한 문장 또는 문서'라는 가정을 활용해 문서를 숫자로 변환   

표 10.1   

그러나 어떤 단어가 많이 나왔다고 해서 문서의 의미를 파악하는 데 크게 도움이 되지 않는 경우가 존재 (조사, 접속사)   

### 10.1.4 TF-IDF
**TF-IDF**(Term Frequency-Inverse Document Frequency) - 앞의 문제를 보완하기 위해 다음 수식을 활용해 많은 문서에 등장하는 단어의 중요도를 작게 만듦   

표 10.2

그러나 이러한 경우 대부분의 수가 0인 벡터가 되는데 이를 '**희소**(sparse)'하다고 함   
희소한 벡터는 의미를 '압축'해서 담고 있지 못하기 때문에 벡터 사이의 관계를 활용하기 어려워 이를 대비한 것을 **밀집 임베딩**(dense embedding)이라고 부름   

### 10.1.5 워드투백
**워드투백**(word2vec) - 단어가 '함께 등장하는 빈도' 정보를 활용해 단어의 의미를 압축하는 단어 임베딩 방법   
1. CBOW(Continuous Bag of Words) - 주변 단어로 가운데 단어를 예측하는 방식
2. 스킵그램(skip-gram) - 중간 단어로 주변 단어를 예측하는 방식   

그림 10.1

이 경우 유사한 벡터가 거리와 방향이 나오기 때문에 단어와 단어 사이의 관계나 의미에 대해 확인할 수 있음   

## 10.2 문장 임베딩 방식
### 10.2.1 문장 사이의 관계를 계산하는 두 가지 방법
BERT(Bidirectional Encoder Representations from Transformers) - 트랜스포머 인코더 구조를 활용하여 입력 문장을 문장 임베딩으로 변환하는데 있어 뛰어난 성능을 보임   
문장과 문장 사이의 관계를 계산하는 방법은 크게 두 가지로 나눌 수 있음   
1. **바이 인코더**(bi-encoder) - 각각의 문장을 입력으로 넣고 모델의 출력 결과인 문장 임베딩 벡터 사이의 유사도를 코사인 유사도와 같은 별도의 계산을 통해 구함   
2. **교차 인코더**(cross-encoder) - 두 문장을 함께 BERT 모델에 입력으로 넣고, 모델이 직접 두 문장 사이의 관계를 0에서 1 사이의 값으로 출력   
**풀링 층** - 문장의 길이가 달라져도 문장 임베딩의 차원이 같도록 맞춰주는 층   

그림 10.3   

교차 인코더는 모든 문장 조합에 대해 유사도를 계산해야 가장 유사한 문장을 검색할 수 있어 확장성이 떨어짐   
바이 인코더는 각 문장의 독립적인 임베딩을 결과로 반환하기 때문에 추가적인 연산이 필요 없음   

그림 10.9

이 때문에 바이 인코더를 활용하면 문장이 많아져도 계산에 오랜 시간이 걸리지 않음   

### 10.2.2 바이 인코더 모델 구조
문장의 길이가 다를 때 서로 다른 개수의 임베딩이 반환된다면 계산하기 어렵기 때문에 풀링 층을 사용해 문장을 대표하는 1개의 임베딩으로 통합   

그림 10.10

풀링 모더의 세 가지 방식

1. 클래스 모드(pooling_mode_cls_tokens): BERT 모델의 첫 번째 토큰인 [CLS] 토큰의 출력 임베딩을 문장 임베딩으로 사용   
2. 평균 모드(pooling_mode_mean_tokens): BERT 모델에서 모든 입력 토큰의 출력 임베딩을 평균한 갑승 문장 임베딩으로 사용   
3. 최대 모드(pooling_mode_max_tokens): BERT 모델의 모든 입력 토큰의 출력 임베딩에서 문장 길이 방향에서 최대값을 찾아 문장 임베딩으로 사용   

세 가지 풀링 모드 중에서는 평균 모드를 일반적으로 많이 활용   

### 10.2.3 Sentence-Transformers로 텍스트와 이미지 임베딩 생성해 보기
```
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

embs = model.encode(['잠이 안 옵니다',
                     '졸음이 옵니다',
                     '기차가 옵니다'])

cos_scores = util.cos_sim(embs, embs)
print(cos_scores)
# tensor([[1.0000, 0.6410, 0.1887],
#         [0.6410, 1.0000, 0.2730],
#         [0.1887, 0.2730, 1.0000]])

허깅페이스 모델 허브에서 제공하는 이미지 모델을 활용하면 이미지도 이미지 임베딩으로 쉽게 변환할 수 있음   
```
### 10.2.4 오픈소스와 상업용 임베딩 모델 비교하기
상업용 모델은 대량의 데이터로 학습된 만큼 성능이 뛰어나고 LLM 텍스트 생성에 비해 훨씬 낮은 비용으로 사용이 가능   
그러나 사용자가 자신의 데이터에 특화된 임베딩 모델을 만을 수 없음   

## 10.3 실습: 의미 검색 구현하기   
의미 검색(semantic search) - 단순히 키워드 매칭을 통한 검색이 아니라 밀집 임베딩을 이용해 문장이나 문서의 의미를 고려한 검색을 수행하는 것   

### 10.3.1 의미 검색 구현하기   
의미 검색은 키워드 검색과 달리 동일한 키워드가 사용되지 않아도 의미적 유사성이 있다면 가깝게 평가함   
그러나 관련성이 떨어지는 검색 결과가 나오기도 함   
```
query = klue_mrc_dataset[3]['question'] # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, 3)

for idx in indices[0]:
  print(klue_mrc_dataset['context'][idx][:50])

# 출력 결과
# 태평양 전쟁 중 뉴기니 방면에서 진공 작전을 실시해 온 더글러스 맥아더 장군을 사령관으로 (오답)
# 태평양 전쟁 중 뉴기니 방면에서 진공 작전을 실시해 온 더글러스 맥아더 장군을 사령관으로 (오답)
# 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)
```
### 10.3.2 라마인덱스에서 Sentence-Transformers 모델 사용하기
```
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
# 로컬 모델 활용하기
# service_context = ServiceContext.from_defaults(embed_model="local")

text_list = klue_mrc_dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

index_llama = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)
```
## 10.4 검색 방식을 조합해 성능 높이기
### 10.4.1 키워드 검색 방식: BM25
**BM25** - TF-IDF와 유사한 통계 기반 스코어링 방법으로, 문서 길이에 대한 가중치를 추가한 알고리즘   
간단하고 계산량이 적으면서도 뛰어난 성능을 보여 대표적인 검색 엔진인 일래스틱서치(Elasticsearch)의 기본 알고리즘으로 사용   

그림 10.13

**avgdl** - 전체 문서의 길이, **k1과 b** - 개발자가 선택할 수 있는 설정값

그림 10.14

**n(q)** - 쿼리 단어의 토큰 q가 등장한 문서의 수, **N** - 전체 문서의 수

그림 10.15

**f(q,D)** - 특정 문서 D에 토큰 q가 등장하는 횟수, **D**는 D문서의 길이, **k1** - 단어 빈도에 대한 포화 효과를 주는 하이퍼파라미터   

표 10.4

### 10.4.2 상호 순위 조합하기
하이브리드 검색을 위해서는 통계 기반 점수와 임베딩 유사도 점수를 합쳐야 하는데 점수마다 분포가 다르기 때문에 두 점수를 그대로 더하면 둘 중 하나의 영향을 더 크게 반영함   
**상호 순위 조합**(Reciprocal Rank Fusion) - 각 점수에서의 순위를 활용해 점수를 산출   

그림 10.17   

## 10.5 실습: 하이브리드 검색 구현하기
### 10.5.1 BM25 구현하기
```
import math
import numpy as np
from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict

class BM25:
  def __init__(self, corpus:List[List[str]], tokenizer:PreTrainedTokenizer):
    self.tokenizer = tokenizer
    self.corpus = corpus
    self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
    self.n_docs = len(self.tokenized_corpus)
    self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
    self.idf = self._calculate_idf()
    self.term_freqs = self._calculate_term_freqs()

  def _calculate_idf(self):
    idf = defaultdict(float)
    for doc in self.tokenized_corpus:
      for token_id in set(doc):
        idf[token_id] += 1
    for token_id, doc_frequency in idf.items():
      idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
    return idf

  def _calculate_term_freqs(self):
    term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
    for i, doc in enumerate(self.tokenized_corpus):
      for token_id in doc:
        term_freqs[i][token_id] += 1
    return term_freqs

  def get_scores(self, query:str, k1:float = 1.2, b:float=0.75):
    query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
    scores = np.zeros(self.n_docs)
    for q in query:
      idf = self.idf[q]
      for i, term_freq in enumerate(self.term_freqs):
        q_frequency = term_freq[q]
        doc_len = len(self.tokenized_corpus[i])
        score_q = idf * (q_frequency * (k1 + 1)) / ((q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
        scores[i] += score_q
    return scores

  def get_top_k(self, query:str, k:int):
    scores = self.get_scores(query)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_k_scores = scores[top_k_indices]
    return top_k_scores, top_k_indices
```
검색 쿼리 문장과 정답 기사 사이의 일치하는 키워드가 적으면 검색되지 않을 수도 있지만 일치하는 키워드를 바탕으로 관련된 기사는 잘 찾음   

### 10.5.2 상호 순위 조합 구현하기
```
from collections import defaultdict

def reciprocal_rank_fusion(rankings:List[List[int]], k=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)
```
### 10.5.3 하이브리드 검색 구현하기
```
def dense_vector_search(query:str, k:int):
  query_embedding = sentence_model.encode([query])
  distances, indices = index.search(query_embedding, k)
  return distances[0], indices[0]

def hybrid_search(query, k=20):
  _, dense_search_ranking = dense_vector_search(query, 100)
  _, bm25_search_ranking = bm25.get_top_k(query, 100)

  results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
  return results
```
하이브리드 검색을 사용하면 의미 검색과 키워드 검색의 단점을 서로 상호보완할 수 있음   

# 11 자신의 데이터에 맞춘 임베딩 모델 만들기: RAG 개선하기
## 11.1 검색 성능을 높이기 위한 두 가지 방법
바이 인코더와 교차 인코더를 결합해서 사용할 수 있음   
1. 먼저 바이 인코더를 사용해 대규모 문서에서 검색 쿼리와 유사한 소수의 문서를 선별
2. 의미 검색을 통해 선별한 소수의 문서는 유사도를 더 정확히 계산할 수 있는 교차 인코더를 사용해 순서대로 재정렬   

그림 11.1   

바이 인코더를 추가 학습하기 때문에 검색 성능을 높일 수 있고 문장 임베딩 모델을 사용하려는 데이터셋으로 추가 학습

## 11.2 언어 모델을 임베딩 모델로 만들기
문장 임베딩 모델은 대량의 텍스트 데이터로 사전 학습한 언어 모델인 첫 번째 층과 입력 문장의 길이에 따라 달라질 수 있는 풀링 층으로 구성
### 11.2.1 대조 학습
**대조 학습**(contrastive learning) - 관련이 있거나 유사한 데이터는 더 가까워지도록 만들고 관련이 없거나 유사하지 않은 데이터는 더 멀어지도록 하는 학습 방식   
### 11.2.2 실습: 학습 준비하기
세 가지 데이터 전처리를 수행
1. 학습 데이터의 일부를 검증을 위한 데이터셋으로 분리
2. 유사도 점수를 0~1 사이로 정규화
3. torch.utils.data.DataLoader를 사용해 배치 데이터로 만듦   
```
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
### 11.2.3 실습: 유사한 문장 데이터로 임베딩 모델 학습하기
```
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
출력 결과를 확인하면 학습 전에 0.364였던 점수가 0.896으로 크게 향상

## 11.3 임베딩 모델 미세 조정하기
### 11.3.1 실습: 학습 준비
기존의 데이터는 많은 필드로 구성되어 있기 때문에 학습을 하는데 필요한 title, question, context를 제외한 나머지 컬럼들을 제거   
```
from datasets import load_dataset
klue_mrc_train = load_dataset('klue', 'mrc', split='train')
klue_mrc_test = load_dataset('klue', 'mrc', split='validation')

df_train = klue_mrc_train.to_pandas()
df_test = klue_mrc_test.to_pandas()

df_train = df_train[['title', 'question', 'context']]
df_test = df_test[['title', 'question', 'context']]  
```
서로 question과 context 칼럼이 서로 관련 있는 경우 label을 1로 지정하고 없는 경우에는 label을 0으로 지정  
``` 
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
### 11.3.2 MNR 손실을 활용해 미세 조정하기
**MNR(Multiple Negatives Ranking) 손실** - 하나의 배치 데이터 안에서 positive 데이터만 존재하는 경우 다른 데이터의 기사 본문을 서로 관련이 없는 negative 데이터로 사용하여 학습   
```
epochs = 1
save_path = './klue_mrc_mnr'

sentence_model.fit(
    train_objectives=[(loader, loss)],
    epochs=epochs,
    warmup_steps=100,
    output_path=save_path,
    show_progress_bar=True
)  
```
## 11.4 검색 품질을 높이는 순위 재정렬
교차 인코더는 관련이 있는 질문-내용 쌍과 관련이 없는 질문-내용 쌍을 구분해야 하기 때문에 학습 데이터셋에 모두 포함돼야 함  
``` 
train_batch_size = 16
num_epochs = 1
model_save_path = 'output/training_mrc'

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

cross_model.fit(
    train_dataloader=train_dataloader,
    epochs=num_epochs,
    warmup_steps=100,
    output_path=model_save_path
)
```
## 11.5 바이 인코더와 교차 인코더로 개선된 RAG 구현하기
아래와 같은 세 가지 케이스로 나눠 비교   
1. 기본 임베딩 모델로 검색하기   
2. 미세 조정한 임베딩 모델로 검색하기   
3. 미세 조정한 모델과 교차 인코더를 결합해 검색하기   
```
def make_question_context_pairs(question_idx, indices):
  return [[klue_mrc_test['question'][question_idx], klue_mrc_test['context'][idx]] for idx in indices]

def rerank_top_k(cross_model, question_idx, indices, k):
  input_examples = make_question_context_pairs(question_idx, indices)
  relevance_scores = cross_model.predict(input_examples)
  reranked_indices = indices[np.argsort(relevance_scores)[::-1]]
  return reranked_indices
```
### 11.5.1 기본 임베딩 모델로 검색하기
```
from sentence_transformers import SentenceTransformer
base_embedding_model = SentenceTransformer('shangrilar/klue-roberta-base-klue-sts')
base_index = make_embedding_index(base_embedding_model, klue_mrc_test['context'])
evaluate_hit_rate(klue_mrc_test, base_embedding_model, base_index, 10)
# (0.88, 13.216430425643921)
```
### 11.5.2 미세 조정한 임베딩 모델로 검색하기
```
finetuned_embedding_model = SentenceTransformer('shangrilar/klue-roberta-base-klue-sts-mrc')
finetuned_index = make_embedding_index(finetuned_embedding_model, klue_mrc_test['context'])
evaluate_hit_rate(klue_mrc_test, finetuned_embedding_model, finetuned_index, 10)
# (0.946, 14.309881687164307)
```
### 11.5.3 미세 조정한 임베딩 모델과 교차 인코더 조합하기
교차 인코더의 경우 속도가 느리기 때문에 전체 문서를 대상으로 검색하지 않고 상위 N개만을 대상으로 계산   
```
hit_rate, cosumed_time, predictions = evaluate_hit_rate_with_rerank(klue_mrc_test, finetuned_embedding_model, cross_model, finetuned_index, bi_k=30, cross_k=10)
hit_rate, cosumed_time
# (0.973, 1103.055629491806)
```
표 11.1   
1.1초는 앞선 시간에 비해 상당히 긴 시간이지만, 모델 경량화 기법을 적용하면 시간을 더 줄일 수 있음   
