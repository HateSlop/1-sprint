# Hateslop Week4

---

웹크롤링 = Web(거미줄) + Crawling(기어다니다)

거미가 마치 거미줄을 움직이듯이 웹의 정보를 긁어 오는 것 = 웹 스크래핑

# 웹 크롤링 활용

- 데이터 분석 과정
- 웹사이트 자동화
- 인공지능 학습 데이터

# 웹 크롤링 주요 활용 사례

- 상품, 컨텐츠 자동 업로드
- 부동산 주식 재테크 데이터 수집
- 인스타그램. 유튜브 모니터링 분석
- 뉴스 데이터 수집
- 논문, 구인광고 데이터 수집

# HTTP 통신

웹브라우저와 웹 서버 사이에 데이터를 주고 받는데 사용되는 통신

- 요청(URL 주소): www.naver.com
- 응답(HTML) : 페이지에 대한 정보가 들어 있음

# 웹사이트 개발의 3요소

## HTML

- Hyper Text Markup Language
- 웹사이트의 구조를 표시하기 위한 언어

### 태그 구조

<태그이름 속성(attribute)=”속성값”> 내용 </태그이름>

부모 태그와 자식 태그 

주석 

<!— 여기부터는 뉴스 기사 영역 —>

<!—<p> 주석 처리하면 어떻게 될까?</p> —>

# CSS 기본 문법

선택자 {속성명: 속성값}

h1 {color : red; }

페이지 안에 있는 모든 h1 태그에 대해 글자색깔을 빨강으로 바꿔라

선택자(selector)

- 웹페이지에서 원하는 태그를 선택하는 문법

- 태그 선택자(태그 이름으로 선택)
- 클래스 선택자 (클래스 속성 값으로 선택하는 것)
    
    . 클래스명
    
- 아이디 선택자 (아이디 속성 값으로 선택하는 것)
    
    # 아이디명
    
- 자식 선택자(바로 아래 자식태그를 선택하는 것)
    
    . header > p 
    

연습 사이트: [https://flukeout.github.io](https://flukeout.github.io)

# 웹크롤링 기초

## 정적 페이지(static page) 크롤링

: 데이터의 추가적인 변경이 일어나지 않는 페이지 

- 데이터 받아오기
    
    파이썬에서 서버에 요청을 보내고 응답받기
    
    HTTP 통신으로 HTML 받아오기
    
- 데이터 뽑아오기
    
    HTML에서 원하는 부분만 추출
    
    CSS 선택자를 잘 만드는 것이 핵심
    

크롤링 연습 사이트: [https://startcoding.pythonanywhere.com/basic](https://startcoding.pythonanywhere.com/basic)  

포레스트 이론

- 숲: 페이지 전체 HTML
- 나무: 원하는 정보를 모두 담는 태그

# URL 조작자

## URL(Uniform Resource Locator)

- 인터넷 주소 형식
- Protocol - Domain - Path - Parameter

https://search.naver.com/search.naver?where=news&query=삼성전자

# 페이징 알고리즘

```python
for i in range(1,?):
	f”https://startcoding.pythonanywhere.com/basic?page={i}”
```

# 동적크롤링

selenium 사용

라이브러리 사용이 계속 바뀌니 아래 카페주소에서 업데이트된 사용법 숙지

https://cafe.naver.com/startcodingofficial