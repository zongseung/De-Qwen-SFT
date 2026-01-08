# 전력수요 보고서 자동 생성 에이전트 시스템

## 📋 시스템 아키텍처

### LangGraph 기반 수퍼바이저 에이전트 시스템

```
사용자 질문/요청
    ↓
┌──────────────────────────────────┐
│   수퍼바이저 에이전트 (중앙 조정)  │
│   - 작업 분석 및 라우팅          │
│   - 하위 에이전트 결과 통합       │
│   - 다음 작업 결정               │
└──────────────────────────────────┘
    ↓              ↓
    ↓         ┌─────────────────────┐
    ↓         │  DB 전문가 에이전트   │
    ↓         │  - 데이터 조회        │
    ↓         │  - CSV/SQL 처리      │
    ↓         │  - 데이터 검증        │
    ↓         └─────────────────────┘
    ↓              ↓
    ↓         (무조건 수퍼바이저 복귀)
    ↓              ↓
    ↓         ┌─────────────────────┐
    └────────→│ 보고서 작성 전문가    │
              │ - SFT 모델 활용      │
              │ - 보고서 생성         │
              │ - 형식 검증           │
              └─────────────────────┘
                   ↓
              수퍼바이저 복귀
                   ↓
              최종 결과 반환
```

---

## 🎯 핵심 원칙

### 1. 수퍼바이저 중심 워크플로우
- 모든 하위 에이전트는 **반드시 수퍼바이저에게 복귀**
- 수퍼바이저만이 다음 작업을 결정
- 하위 에이전트 간 직접 통신 금지

### 2. 단방향 워크플로우
```
수퍼바이저 → 하위 에이전트 → 수퍼바이저 → 다음 작업
```

### 3. 상태 관리
- LangGraph State를 통한 모든 정보 공유
- 각 에이전트는 State만 읽고 업데이트

---

## 🏗️ 에이전트 상세 설계

### 1. 수퍼바이저 에이전트 (Supervisor)

**역할:**
- 사용자 요청 분석
- 적절한 하위 에이전트 선택
- 하위 에이전트 결과 평가
- 다음 작업 결정 (계속/완료)

**판단 로직:**
```python
if "데이터 조회" or "CSV" or "통계" in 요청:
    → DB 전문가 에이전트
elif "보고서 작성" in State and 데이터 있음:
    → 보고서 작성 전문가
elif 모든 작업 완료:
    → FINISH
```

**도구:**
- LLM (GPT-4 또는 Claude) - 판단용
- State 읽기/쓰기

---

### 2. DB 전문가 에이전트 (Database Expert)

**역할:**
- 전력수요 데이터 조회
- CSV 파일 읽기/파싱
- SQL 쿼리 실행 (필요 시)
- 데이터 검증 및 전처리

**도구:**
- `pandas` - CSV 처리
- `sqlite3` - SQL 쿼리 (선택)
- 데이터 검증 함수

**입력 (State):**
```python
{
    "supervisor_request": "2025년 3월 데이터 조회",
    "csv_path": "sample_power_demand_2025.csv",
    "filters": {"year": 2025, "month": 3}
}
```

**출력 (State 업데이트):**
```python
{
    "db_result": {
        "year": 2025,
        "month": 3,
        "max_load": 9200,
        "avg_load": 7100,
        "yoy_change": 2.3,
        "temperature": 15.2,
        "precipitation": 45.3
    },
    "next_agent": "supervisor"  # 무조건 수퍼바이저 복귀
}
```

**중요 규칙:**
- ✅ 데이터 조회 후 **무조건 수퍼바이저 복귀**
- ❌ 보고서 작성 에이전트 직접 호출 금지

---

### 3. 보고서 작성 전문가 (Report Writer Expert)

**역할:**
- SFT 학습된 모델 활용
- 전문적인 전력수요 보고서 생성
- 보고서 형식 검증

**도구:**
- SFT 모델 (`power_demand_sft_model`)
- 보고서 템플릿
- 형식 검증기

**입력 (State):**
```python
{
    "supervisor_request": "2025년 3월 보고서 작성",
    "db_result": {  # DB 전문가가 제공한 데이터
        "year": 2025,
        "month": 3,
        "max_load": 9200,
        ...
    },
    "report_type": "full"  # 또는 "summary"
}
```

**출력 (State 업데이트):**
```python
{
    "report": "# 2025년 3월 전력수요 분석 보고서\n\n...",
    "report_metadata": {
        "word_count": 850,
        "sections": 4,
        "generated_at": "2025-01-08 12:00:00"
    },
    "next_agent": "supervisor"  # 수퍼바이저 복귀
}
```

---

## 📊 State 스키마

### LangGraph State 정의

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import MessagesState

class PowerDemandState(TypedDict):
    """전력수요 에이전트 시스템 State"""

    # 사용자 입력
    messages: Annotated[list, "사용자와의 대화 기록"]
    user_request: str  # 원본 요청

    # 수퍼바이저
    next_agent: Literal["db_expert", "report_writer", "FINISH"]
    supervisor_reasoning: str  # 판단 근거

    # DB 전문가
    csv_path: str | None
    query_filters: dict | None
    db_result: dict | None  # 조회된 데이터

    # 보고서 작성 전문가
    report_type: Literal["full", "summary"] | None
    report: str | None  # 생성된 보고서
    report_metadata: dict | None

    # 최종 결과
    final_output: str | None
```

---

## 🔄 워크플로우 시나리오

### 시나리오 1: 단순 보고서 생성

**사용자 요청:**
```
"2025년 3월 전력수요 보고서를 작성해주세요."
```

**워크플로우:**
```
1. 수퍼바이저
   - 요청 분석: "데이터 필요 + 보고서 작성"
   - 결정: next_agent = "db_expert"

2. DB 전문가
   - sample_power_demand_2025.csv에서 2025년 3월 데이터 조회
   - State 업데이트: db_result = {...}
   - 복귀: next_agent = "supervisor"

3. 수퍼바이저
   - 상황 평가: 데이터 있음, 보고서 작성 필요
   - 결정: next_agent = "report_writer"

4. 보고서 작성 전문가
   - SFT 모델로 보고서 생성
   - State 업데이트: report = "# 2025년 3월..."
   - 복귀: next_agent = "supervisor"

5. 수퍼바이저
   - 상황 평가: 모든 작업 완료
   - 결정: next_agent = "FINISH"

6. 최종 결과 반환
```

---

### 시나리오 2: 복잡한 요청 (여러 월 비교)

**사용자 요청:**
```
"2025년 1월부터 5월까지 전력수요 추이를 분석한 보고서를 작성해주세요."
```

**워크플로우:**
```
1. 수퍼바이저
   - 요청 분석: "여러 월 데이터 필요"
   - 결정: next_agent = "db_expert"

2. DB 전문가
   - 1월~5월 데이터 모두 조회
   - State 업데이트: db_result = [1월, 2월, 3월, 4월, 5월]
   - 복귀: next_agent = "supervisor"

3. 수퍼바이저
   - 상황 평가: 데이터 충분, 비교 분석 보고서 필요
   - 결정: next_agent = "report_writer"

4. 보고서 작성 전문가
   - SFT 모델로 추이 분석 보고서 생성
   - State 업데이트: report = "# 2025년 1-5월 추이 분석..."
   - 복귀: next_agent = "supervisor"

5. 수퍼바이저
   - 결정: next_agent = "FINISH"
```

---

### 시나리오 3: 데이터 부족 시 재시도

**사용자 요청:**
```
"2026년 1월 보고서를 작성해주세요."
```

**워크플로우:**
```
1. 수퍼바이저 → db_expert

2. DB 전문가
   - 2026년 데이터 조회 시도
   - 결과: 데이터 없음
   - State 업데이트: db_result = None, error = "데이터 없음"
   - 복귀: next_agent = "supervisor"

3. 수퍼바이저
   - 상황 평가: 데이터 없음, 보고서 작성 불가
   - 결정: next_agent = "FINISH"
   - final_output = "죄송합니다. 2026년 1월 데이터가 없습니다."
```

---

## 🛠️ 구현 가이드

### 1. 디렉토리 구조

```
de-llama/
├── agents/
│   ├── __init__.py
│   ├── supervisor.py          # 수퍼바이저 에이전트
│   ├── db_expert.py           # DB 전문가 에이전트
│   └── report_writer.py       # 보고서 작성 전문가
├── graph/
│   ├── __init__.py
│   ├── state.py               # State 정의
│   └── workflow.py            # LangGraph 워크플로우
├── tools/
│   ├── __init__.py
│   ├── csv_loader.py          # CSV 처리 도구
│   └── report_generator.py    # SFT 모델 래퍼
├── power_demand_sft_model/    # 학습된 SFT 모델
├── data/
│   └── sample_power_demand_2025.csv
├── main.py                    # 실행 엔트리포인트
└── README_AGENT_SYSTEM.md     # 이 문서
```

---

### 2. 필수 라이브러리

```bash
# pyproject.toml에 추가
dependencies = [
    # 기존 SFT 라이브러리
    "transformers>=4.57.3",
    "torch>=2.9.1",
    "peft>=0.18.0",

    # LangGraph & LangChain
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",  # 또는 langchain-anthropic

    # 데이터 처리
    "pandas>=2.0.0",
    "sqlite3",  # Python 내장

    # 유틸리티
    "python-dotenv>=1.0.0",
]
```

---

### 3. 구현 순서

#### Step 1: State 정의
```python
# graph/state.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class PowerDemandState(TypedDict):
    messages: Annotated[list, add_messages]
    user_request: str
    next_agent: Literal["db_expert", "report_writer", "FINISH"]
    # ... (위 스키마 참조)
```

#### Step 2: 각 에이전트 구현
```python
# agents/supervisor.py
def supervisor_agent(state: PowerDemandState) -> PowerDemandState:
    """수퍼바이저 에이전트"""
    # LLM으로 판단
    # next_agent 결정
    return updated_state

# agents/db_expert.py
def db_expert_agent(state: PowerDemandState) -> PowerDemandState:
    """DB 전문가 에이전트"""
    # 데이터 조회
    # 무조건 next_agent = "supervisor"
    return updated_state

# agents/report_writer.py
def report_writer_agent(state: PowerDemandState) -> PowerDemandState:
    """보고서 작성 전문가"""
    # SFT 모델로 보고서 생성
    # next_agent = "supervisor"
    return updated_state
```

#### Step 3: LangGraph 워크플로우 구성
```python
# graph/workflow.py
from langgraph.graph import StateGraph, END

def create_workflow():
    workflow = StateGraph(PowerDemandState)

    # 노드 추가
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("db_expert", db_expert_agent)
    workflow.add_node("report_writer", report_writer_agent)

    # 엣지 추가 (라우팅)
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,  # 라우팅 함수
        {
            "db_expert": "db_expert",
            "report_writer": "report_writer",
            "FINISH": END
        }
    )

    # 하위 에이전트는 무조건 수퍼바이저 복귀
    workflow.add_edge("db_expert", "supervisor")
    workflow.add_edge("report_writer", "supervisor")

    # 시작점
    workflow.set_entry_point("supervisor")

    return workflow.compile()
```

#### Step 4: 라우팅 함수
```python
def route_supervisor(state: PowerDemandState) -> str:
    """수퍼바이저 라우팅 결정"""
    return state["next_agent"]
```

#### Step 5: 실행
```python
# main.py
from graph.workflow import create_workflow

def main():
    app = create_workflow()

    # 초기 State
    initial_state = {
        "messages": [],
        "user_request": "2025년 3월 전력수요 보고서를 작성해주세요.",
        "next_agent": "supervisor",
        # ... 나머지 None
    }

    # 실행
    result = app.invoke(initial_state)

    print(result["final_output"])
```

---

## 🔐 환경 변수 설정

```bash
# .env 파일
OPENAI_API_KEY=sk-...           # 수퍼바이저용 LLM
SFT_MODEL_PATH=./power_demand_sft_model
CSV_DATA_PATH=./data/sample_power_demand_2025.csv
```

---

## 📝 에이전트별 프롬프트

### 수퍼바이저 에이전트 프롬프트
```
당신은 전력수요 보고서 생성 시스템의 수퍼바이저입니다.

**역할:**
- 사용자 요청 분석
- 적절한 하위 에이전트 선택 (db_expert, report_writer)
- 작업 완료 판단

**규칙:**
1. 데이터 조회 필요 시 → db_expert
2. 보고서 작성 필요 시 (데이터 있을 때) → report_writer
3. 모든 작업 완료 시 → FINISH

**현재 상황:**
{state_summary}

**다음 에이전트를 선택하세요:**
- db_expert
- report_writer
- FINISH
```

### DB 전문가 프롬프트
```
당신은 전력수요 데이터 조회 전문가입니다.

**작업:**
{supervisor_request}

**도구:**
- CSV 파일 읽기
- 데이터 필터링
- 통계 계산

**반환 형식:**
JSON 형태로 조회 결과 반환

**중요:** 작업 완료 후 무조건 수퍼바이저에게 복귀하세요.
```

---

## 🧪 테스트 케이스

### 테스트 1: 단순 보고서 생성
```python
input = "2025년 3월 보고서 작성"
expected_flow = ["supervisor", "db_expert", "supervisor", "report_writer", "supervisor", "FINISH"]
```

### 테스트 2: 여러 월 비교
```python
input = "2025년 1-5월 추이 분석"
expected_flow = ["supervisor", "db_expert", "supervisor", "report_writer", "supervisor", "FINISH"]
```

### 테스트 3: 데이터 없음
```python
input = "2030년 데이터 조회"
expected_flow = ["supervisor", "db_expert", "supervisor", "FINISH"]
expected_output = "데이터 없음 메시지"
```

---

## 🚀 실행 방법

### CLI 실행
```bash
# 단일 보고서 생성
python main.py --request "2025년 3월 보고서 작성"

# 여러 월 비교
python main.py --request "2025년 1-5월 추이 분석"

# 인터랙티브 모드
python main.py --interactive
```

### API 서버 (FastAPI)
```bash
# 서버 시작
uvicorn api:app --reload

# 요청
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"request": "2025년 3월 보고서 작성"}'
```

---

## 📊 성능 목표

- **응답 시간:** < 30초 (단일 보고서)
- **정확도:** 데이터 조회 100%, 보고서 형식 준수 95%+
- **가용성:** 99%+

---

## 🔄 확장 계획

### Phase 2: 추가 에이전트
- **데이터 시각화 전문가:** 그래프/차트 생성
- **검증 전문가:** 보고서 검증 및 수정 제안

### Phase 3: 멀티모달
- 이미지/차트 포함 보고서
- PDF 자동 생성

---

## 📚 참고 자료

- [LangGraph 공식 문서](https://python.langchain.com/docs/langgraph)
- [Multi-Agent 패턴](https://python.langchain.com/docs/langgraph/how-tos/agent-supervisor)

---

**이 README를 따라 구현하세요. 모든 구현은 이 문서 기준을 준수해야 합니다.** ✅
