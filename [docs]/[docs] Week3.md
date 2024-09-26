# Hateslop Week3

# 12 벡터 데이터베이스로 확장하기: RAG 구현하기

---

```python
!pip install pinecone-client sentence-transformers==2.7.0 datasets==2.19.0 faiss-cpu==1.8.0 transformers openai==1.9.0 -qqq
```

### 12.1 벡터 데이터베이스란

![스크린샷 2024-09-22 16.14.55.png](%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-22_16.14.55.png)

### 12.2 벡터 데이터베이스 작동 원리

### 12.3 실습: HNSW 인덱스의 핵심 파라미터 이해하기

### 12.4 실습: 파인콘으로 벡터 검색 구현하기

### 12.5 실습: 파인콘을 활용해 멀티 모달 검색 구현하기

### 12.6 정리

# 13 LLM 운영하기 (백재현 김민우 발표)

---

LLMOps: LLM 서비스 통합 운영 관리 체제

MLOps  + LLM = LLMOps

## 13.1 MLOps

MLOps : Machine Learning Operations 

![스크린샷 2024-09-22 16.51.26.png](%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-22_16.51.26.png)

모델의 재현성: 이전에 수행했던 ML workflow를  그대로 반복했을 때 동일한 모델을 얻을 수 있는지 여부

### 13.1.1 데이터 관리

### 13.1.2 실험 관리

### 13.1.3 모델 저장소

### 13.1.4 모델 모니터링

## 13.2 LLMOps는 무엇이 다를까?

### 13.2.1 상업용 모델과 오픈소스 모델 선택하기

### 13.2.2 모델 최적화 방법의 변화

### 13.2.3 LLM 평가의 어려움

## 13.3 LLM 평가하기

### 13.3.1 정량적 지표

번역: BLUE (Bilingual Evaluation Understudy Score)

요약이나 번역 등 자연어 생성 모델: Rouge(Recall Oriented Understudy for Gisting Evaluation)

PPL(perplexity): 모델이 새로운 단어를 생성할 때의 불확실성

### 13.3.2 벤치마크 데이터셋을 활용한 평가

### 13.2.3 사람이 직접 평가하는 방식

### 13.3.4 LLM을 통한 평가

### 13.3.5 RAG 평가

- 신뢰성(Faithfulness score)
- 답변 관련성(Answer Relevancy)
- 맥락 관련성(Context Relevancy)

Ragas(RAG Assesment)

 

## 13.4 정리

# 4부 멀티 모달, 에이전트 그리고 LLM의 미래

---

# 14 멀티 모달 LLM

---

```python
!pip install transformers==4.40.1 -qqq
```

## 14.1 멀티 모달 LLM이란

멀티 모달 LLM 이란, 텍스트뿐만 아니라 이미지, 비디오, 오디오, 3D 등 다양한 형식의 데이터를 이해하고 생성할 수 있는 LLM을 말한다.

- 모달리티 인코더
- 입력 프로젝터
- 출력 프로젝터
- 모달리티 생성기

이미지와 텍스트 쌍과 같은 대규모 멀티 모달 데이터 세트로 학습

## 14.2 이미지와 텍스트를 연결하는 모델: CLIP

OpenAI 에서 개발한 CLIP

데이터셋: MS-COCO, Visual Genome, YFCC100M

제로샷 추론: 사전 학습 데이터 이외에 특정 작업을 위한 데이터로 미세 조정하지 않은 상태에서 추론을 수행하는 것을 말함.

```python
# 14.1 허깅페이스로 CLIP 모델 활용

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
```

```python
# 14.2 CLIP 모델 추론

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

디퓨전 모델: 완전한 노이즈 상태의 이미지에서 노이즈를 예측하고 예측된 노이즈를 제거하면서 점차 완전한 노이즈에서 의미있는 이미지를 생성하는 방식

## 14.4 LLaVA

이미지를 인식하는 CLIP 모델과 LLM을 결합해 모델이 이미지를 인식하고 그 이미지에 대한 텍스트를 생성할 수 있다. 

## 14.5 정리

# 15 LLM 에이전트

---

AutoGPT는 달성하고자 하는 목표만 입력하면 내부적으로 LLM이 알아서 어떤 작업이 필요한지를 계획하고 검색, 계산기, 코드 등 다양한 도구를 활용해 목표를 달성하도록 자동하는 프로그램이다. 

```python
!pip install "pyautogen[retrievechat]==0.2.6" -qqq
```

## 15.1 에이전트란

에이전트의 구성 요소

- 감각
- 두뇌
- 행동

## 15.2 에이전트 시스템의 형태

- 단일 에이전트
    
    ```python
    # 예시 프롬프트
    
    You are Story-GPT, an AI designed to autonomously write stories.
    Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.
    
    GOALS:
    1. write a short story about flowers
    
    Constraints:
    1. 4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
    2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
    3. No user assistance
    4. Exclusively use the commands listed in double quotes e.g. "command name"
    
    Commands:
    1. Google Search: "google", args: "input": "<search>"
    2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
    3. Start GPT Agent: "start_agent", args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
    4. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
    5. List GPT Agents: "list_agents", args:
    6. Delete GPT Agent: "delete_agent", args: "key": "<key>"
    7. Clone Repository: "clone_repository", args: "repository_url": "<url>", "clone_path": "<directory>"
    8. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
    9. Read file: "read_file", args: "file": "<file>"
    10. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
    11. Delete file: "delete_file", args: "file": "<file>"
    12. Search Files: "search_files", args: "directory": "<directory>"
    13. Evaluate Code: "evaluate_code", args: "code": "<full_code_string>"
    14. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
    15. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
    16. Execute Python File: "execute_python_file", args: "file": "<file>"
    17. Generate Image: "generate_image", args: "prompt": "<prompt>"
    18. Send Tweet: "send_tweet", args: "text": "<text>"
    19. Do Nothing: "do_nothing", args:
    20. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"
    
    Resources:
    1. Internet access for searches and information gathering.
    2. Long Term memory management.
    3. GPT-3.5 powered Agents for delegation of simple tasks.
    4. File output.
    
    Performance Evaluation:
    1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
    2. Constructively self-criticize your big-picture behavior constantly.
    3. Reflect on past decisions and strategies to refine your approach.
    4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
    
    You should only respond in JSON format as described below 
    Response Format: 
    {
        "thoughts": {
            "text": "thought",
            "reasoning": "reasoning",
            "plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
            "speak": "thoughts summary to say to user",
        },
        "command": {"name": "command name", "args": {"arg name": "value"}},
    }
    
    Ensure the response can be parsed by Python json.loads
    ```
    

- 사용자와 에이전트의 상호작용
- 멀티 에이전트

## 15.3 에이전트 평가하기

에이전트를 평가하는 네 가지 기준

- 유용성
- 사회성
- 가치관
- 진화 능력

## 15.4 실습: 에이전트 구현

```python
import json

openai_api_key = "자신의 API 키 입력"

with open('OAI_CONFIG_LIST.json', 'w') as f:
  config_list = [
    {
        "model": "gpt-4-turbo-preview",
        "api_key": openai_api_key
    },
    {
        "model": "gpt-4o",
        "api_key": openai_api_key,
    },
    {
        "model": "dall-e-3",
        "api_key": openai_api_key,
    }
]
  json.dump(config_list, f)
```

```python
import autogen

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    file_location=".",
    filter_dict={
        "model": ["gpt-4-turbo-preview"],
    },
)

llm_config = {
    "config_list": config_list,
    "temperature": 0,
}
```

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy",
  is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
  human_input_mode="NEVER",
  code_execution_config={"work_dir": "coding", "use_docker": False})
```

```python

user_proxy.initiate_chat(assistant, message="""
삼성전자의 지난 3개월 주식 가격 그래프를 그려서 samsung_stock_price.png 파일로 저장해줘.
plotly 라이브러리를 사용하고 그래프 아래를 투명한 녹색으로 채워줘.
값을 잘 확인할 수 있도록 y축은 구간 최소값에서 시작하도록 해줘.
이미지 비율은 보기 좋게 적절히 설정해줘.
""")
```

```python
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
# AutoGen은 여러 에이전트가 상호 대화하여 작업을 해결할 수 있는 LLM(Large Language Model) 애플리케이션 개발을 가능하게 하는 프레임워크입니다. AutoGen 에이전트는 사용자 정의 가능하며, 대화 가능하고, 인간 참여를 원활하게 허용합니다. LLM, 인간 입력, 도구의 조합을 사용하는 다양한 모드에서 작동할 수 있습니다.
```

```python
assistant.reset()
userproxyagent = autogen.UserProxyAgent(
    name="userproxyagent",
)
userproxyagent.initiate_chat(assistant, message="Autogen이 뭐야?")

# assistant (to userproxyagent):
# "Autogen"은 자동 생성을 의미하는 용어로, 주로 컴퓨터 프로그래밍에서 사용됩니다. 이는 코드, 문서, 또는 다른 데이터를 자동으로 생성하는 프로세스를 가리킵니다. 이는 반복적인 작업을 줄이고, 효율성을 높이며, 오류를 줄일 수 있습니다. 특정 컨텍스트에 따라 "Autogen"의 정확한 의미는 다를 수 있습니다.
```

```python
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    retrieve_config={
        "task": "qa",
        "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        "embedding_function": openai_ef,
        "collection_name": "openai-embedding-3",
    },
)

assistant.reset()
ragproxyagent.initiate_chat(assistant, problem="Autogen이 뭐야?")

# assistant (to ragproxyagent):
# AutoGen은 여러 에이전트가 상호 대화하여 작업을 해결할 수 있는 LLM(Large Language Model) 애플리케이션 개발을 가능하게 하는 프레임워크입니다. AutoGen 에이전트는 사용자 정의 가능하며, 대화 가능하고, 인간 참여를 원활하게 허용합니다. LLM, 인간 입력, 도구의 조합을 사용하는 다양한 모드에서 작동할 수 있습니다.
```

```python
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

```python
def _reset_agents():
    user.reset()
    user_rag.reset()
    coder.reset()
    pm.reset()

def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[user_rag, coder, pm],
        messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_rag.initiate_chat(
        manager,
        problem=PROBLEM,
    )

def norag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[user, coder, pm],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user.initiate_chat(
        manager,
        message=PROBLEM,
    )
```

```python
norag_chat()
# AutoGen Studio는 자동화된 코드 생성 도구입니다. 이 도구를 사용하면 개발자들이 더 빠르게, 더 효율적으로 코드를 작성할 수 있습니다.
# AutoGen Studio를 사용하면 다양한 유형의 소프트웨어 제품을 만들 수 있습니다. 예를 들어, 웹 애플리케이션, 모바일 애플리케이션, 데스크톱 애플리케이션, API, 데이터베이스 등을 만들 수 있습니다.
# ...
rag_chat()
# AutoGen Studio는 AutoGen 프레임워크를 기반으로 한 AI 앱입니다. 이 앱은 AI 에이전트를 빠르게 프로토타입화하고, 스킬을 향상시키고, 워크플로우로 구성하고, 작업을 완료하기 위해 그들과 상호 작용하는 데 도움을 줍니다. 이 앱은 GitHub의 [microsoft/autogen](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio)에서 코드를 찾을 수 있습니다.
# AutoGen Studio를 사용하면 다음과 같은 기능을 수행할 수 있습니다:
# - 에이전트를 구축/구성하고, 그들의 구성(예: 스킬, 온도, 모델, 에이전트 시스템 메시지, 모델 등)을 수정하고, 워크플로우로 구성합니다.
# ...
```

```python
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import PIL
import requests
from openai import OpenAI
from PIL import Image

from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

config_list_4o = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["gpt-4o"],
    },
)

config_list_dalle = autogen.config_list_from_json(
    "OAI_CONFIG_LIST.json",
    filter_dict={
        "model": ["dall-e-3"],
    },
)
```

```python
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

```python
painter = DALLEAgent(name="Painter", llm_config={"config_list": config_list_dalle})

user_proxy = UserProxyAgent(
    name="User_proxy", system_message="A human admin.", human_input_mode="NEVER", max_consecutive_auto_reply=0
)

# 이미지 생성 작업 실행하기
user_proxy.initiate_chat(
    painter,
    message="갈색의 털을 가진 귀여운 강아지를 그려줘",
)
```

```python
image_agent = MultimodalConversableAgent(
    name="image-explainer",
    system_message="Explane input image for painter to create similar image.",
    max_consecutive_auto_reply=10,
    llm_config={"config_list": config_list_4o, "temperature": 0.5, "max_tokens": 1500},
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False
)

groupchat = autogen.GroupChat(agents=[user_proxy, image_agent, painter], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
```

```python
user_proxy.initiate_chat(
    manager,
    message=f"""아래 이미지랑 비슷한 이미지를 만들어줘.
<img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
)
```

```python
user_proxy.initiate_chat(
    manager,
    message="갈색의 털을 가진 귀여운 강아지를 그려줘",
)
```

## 15.5 정리

# 16 새로운 아키텍쳐

---

## 16.1 기존 아키텍쳐의 장단점

![스크린샷 2024-09-24 18.27.50.png](%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-09-24_18.27.50.png)

## 16.2 SSM

SSM: 내부 상태를 가지고 시간에 가지고 시간에 따라 달라지는 시스템을 해석하기 위해 사용되는 모델링 방법을 말함.

## 16.3 선택 메커니즘

## 16.4 맘바

## 16.5 코드로 보는 맘바