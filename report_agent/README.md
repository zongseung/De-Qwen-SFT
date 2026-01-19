# Power Demand Report Agent

전력수요(수요/기상) 데이터와 예측 모델(LSTM/ARIMA/Holt-Winters), 그리고 SFT 튜닝된 LLM(vLLM)을 결합해 **월간 전력수요 전망 보고서(Markdown)**를 자동 생성합니다.

이 README는 “어떤 노드(컴포넌트)가 어떤 책임을 가지고, 어떤 순서로, 어떤 데이터 형태로 상호작용하는지”를 **코드 기준으로 완전히 구체화**한 문서입니다.

---

## 0) 설계 원칙 (역할 분리)

- **정답이 필요한 것**(수치/표/차트/예측)은 코드 기반 **Tools(MCP)**가 만든다: DB 조회, YoY 계산, 예측, 차트 렌더링
- **서술이 필요한 것**(문장/요약/설명)은 **LLM(vLLM)**이 만든다
- **품질 보증**은 CLI 후처리가 담당한다: 반복 제거, 표 형식 강제, 차트 링크 삽입

---

## 1) 아키텍처 (Mermaid node graph)

### 1.1 전체 구성도 (한 장으로 보는 노드)

```mermaid
flowchart TB
  User[User] --> CLI["generate_report.py (CLI 오케스트레이터)"]

  subgraph MCP_Layer["MCP Layer (데이터/예측/차트)"]
    direction TB
    subgraph MCP_Servers["MCP 서버 (택1)"]
      MCP_PURE["mcp_pure_server.py\n(FastMCP, stdio)\n자동 subprocess"]
      MCP_HTTP["server.py\n(FastAPI, :8001)\n별도 실행 필요"]
    end
    Tools["tools.py\n- CombinedTools\n- ForecastTools\n- ChartTools"]
    DB["demand.db\n(SQLite)"]
    Models["artifacts/\n- models/*.pth\n- scalers/*.pkl"]
  end

  subgraph LLM_Layer["LLM Layer (서술 생성, 선택)"]
    direction LR
    vLLM["vLLM Server (:8000)\nserve_vllm.py\n별도 실행 필요"]
    SFT["power_demand_merged_model_llama3/"]
    vLLM --> SFT
  end

  CLI -- "1. --mcp-mode stdio (권장)\nsubprocess 자동 실행" --> MCP_PURE
  CLI -. "1. --mcp-mode http\nPOST /tools/*" .-> MCP_HTTP

  MCP_PURE --> Tools
  MCP_HTTP --> Tools
  Tools --> DB
  Tools --> Models

  CLI -- "2. --llm-url (선택)\nPOST /v1/completions" --> vLLM

  CLI --> Output["reports/\n- report_*.md\n- charts/*.png"]

  style MCP_PURE fill:#90EE90
  style vLLM fill:#FFE4B5
```

### 1.2 보고서 생성 시퀀스 (정확한 실행 순서)

> 실제 구현 기준(파일/함수명): `report_agent/generate_report.py`의 HTTP 모드와 stdio 모드가 거의 동일한 순서로 동작합니다.

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as generate_report.py
  participant MCP as MCP Server (HTTP or stdio)
  participant Tools as tools.py (CombinedTools/ChartTools)
  participant DB as demand.db
  participant Models as LSTM/ARIMA Models
  participant LLM as vLLM (:8000)
  participant FS as filesystem (reports/)

  U->>CLI: run CLI args (--year, --month, --mcp-mode, --llm-url)

  Note over CLI,Tools: Step 1: 데이터 조회
  CLI->>MCP: get_report_data(year, month)
  MCP->>Tools: CombinedTools.get_report_data()
  Tools->>DB: SELECT demand/weather
  DB-->>Tools: rows
  Tools-->>MCP: summary + weekly + historical + weather + monthly_forecast
  MCP-->>CLI: JSON response

  Note over CLI,Models: Step 2: 주차별 예측
  CLI->>MCP: forecast_weekly_demand(year, month, model, include_next_month)
  MCP->>Tools: ForecastTools.forecast_weekly()
  Tools->>DB: SELECT historical data (12주)
  Tools->>Models: LSTM inference (*.pth)
  Models-->>Tools: predictions
  Tools-->>MCP: {model, forecasts[], weeks_count}
  MCP-->>CLI: JSON response

  Note over CLI,FS: Step 3: 차트 생성
  CLI->>MCP: generate_yearly_monthly_chart(...)
  MCP->>Tools: ChartTools.generate_yearly_monthly_chart()
  Tools->>DB: SELECT yearly data
  Tools-->>MCP: (HTTP: filepath) or (stdio: image_base64)
  MCP-->>CLI: chart response
  CLI->>FS: write charts/*.png (if base64)

  Note over CLI: Step 4: 프롬프트 생성
  CLI->>CLI: build_report_prompt(data, forecast)

  Note over CLI,LLM: Step 5: LLM 호출 (선택)
  alt --llm-url provided
    CLI->>LLM: POST /v1/completions (prompt)
    LLM-->>CLI: raw text
    CLI->>CLI: remove_repetitions()
    CLI->>CLI: apply_table_fixes()
    CLI->>CLI: insert_chart_after_section2()
    CLI->>FS: save report_*_llm_*.md
  else no --llm-url
    CLI->>FS: save report_*_prompt_*.md (프롬프트만 저장)
  end
```

---

## 2) 실행 모드 (MCP 통신 방식)

### 2.1 MCP stdio 모드 (권장)

- CLI가 `report_agent/mcp_server/mcp_pure_server.py`를 **subprocess로 띄우고**, 표준입출력(stdio)으로 MCP 세션을 맺습니다.
- 네트워크 포트를 열지 않아도 되고(충돌/방화벽 이슈 감소), “한 번 실행해서 한 번 보고서 생성”에 적합합니다.
- 구현 파일:
  - 클라이언트: `report_agent/mcp_client_pure.py` (`ClientSession` + `mcp.client.stdio`)
  - 서버: `report_agent/mcp_server/mcp_pure_server.py` (`FastMCP`)

### 2.2 MCP HTTP 모드

- FastAPI 서버(`:8001`)를 별도 프로세스로 띄우고 CLI가 HTTP로 `/tools/*` 엔드포인트를 호출합니다.
- 구현 파일:
  - 클라이언트: `report_agent/mcp_client.py` (HTTP POST)
  - 서버: `report_agent/mcp_server/server.py` (FastAPI + uvicorn)

### 2.3 (참고) Pure MCP SSE 모드

- `mcp_client_pure.py`는 SSE도 지원하지만, `generate_report.py`는 현재 stdio만 사용합니다.
- 필요 시 확장 포인트: `MCPClientPure(mode="sse", server_url="http://localhost:8001/sse")`

---

## 3) 빠른 시작 (정확한 커맨드)

### 3.1 (선택) vLLM 서버 실행 (LLM로 “보고서 본문”까지 생성할 때)

```bash
cd /root/De-Qwen-SFT
python serve_vllm.py --mode server --host 0.0.0.0 --port 8000
```

`report_agent/generate_report.py`는 기본적으로 `POST {llm_url}/v1/completions`를 호출합니다.

### 3.2 MCP stdio 모드 (권장)

```bash
cd /root/De-Qwen-SFT/report_agent

# 1) 프롬프트만 저장 (LLM 없이)
python generate_report.py --year 2025 --month 11 --mcp-mode stdio

# 2) LLM까지 포함해 보고서 저장
python generate_report.py --year 2025 --month 11 --mcp-mode stdio --llm-url http://localhost:8000

# 3) 다음달까지 예측(8주) 포함
python generate_report.py --year 2025 --month 11 --mcp-mode stdio --include-next-month --llm-url http://localhost:8000
```

### 3.3 MCP HTTP 모드

```bash
# 터미널 1: MCP 서버 실행
cd /root/De-Qwen-SFT/report_agent
python -m mcp_server.server --port 8001

# 터미널 2: 보고서 생성
cd /root/De-Qwen-SFT/report_agent
python generate_report.py --year 2025 --month 11 --mcp-mode http --mcp-url http://localhost:8001 --llm-url http://localhost:8000
```

---

## 4) CLI 옵션 (generate_report.py)

```bash
python generate_report.py --year YYYY --month M [--mcp-mode http|stdio] [--mcp-url ...] [--llm-url ...] [--forecast-model ...] [--include-next-month] [--output ...] [--json] [--prompt-only]
```

| 옵션 | 의미 | 기본값/비고 |
|---|---|---|
| `--year` | 대상 연도 | 필수 |
| `--month` | 대상 월(1-12) | 필수 |
| `--mcp-mode` | `http` 또는 `stdio` | 기본 `http` |
| `--mcp-url` | HTTP 모드 MCP 서버 URL | 기본 `http://localhost:8001` |
| `--llm-url` | vLLM URL(없으면 프롬프트만 저장) | 없음 |
| `--forecast-model` | `lstm`, `arima`, `holt_winters`, `ensemble` | 기본 `lstm` |
| `--include-next-month` | 다음달까지 예측(8주) | 기본 `False` |
| `--output` | 출력 디렉토리 | 기본 `./reports` |
| `--json` | 보고서 대신 JSON 데이터 출력 | 디버그용 |
| `--prompt-only` | 프롬프트만 stdout 출력 | 파일 저장 안 함 |

---

## 5) 컴포넌트별 구성/동작 (파일 단위로 “무슨 일을 하는지”)

### 5.1 오케스트레이터 (CLI)

- `report_agent/generate_report.py`
  - 입력(Args): `year/month`, `mcp-mode`, `forecast-model`, `include-next-month`, `llm-url`, `output`
  - MCP에서 가져오는 “정답 데이터”:
    - `get_report_data(year, month)` → 요약/주차/과거/기상/월예측 등 패키지
    - `forecast_weekly_demand(year, month, model, include_next_month)` → 주차별 최대수요 예측
    - `generate_yearly_monthly_chart(target_year, target_month, years, output_dir?)` → 차트(PNG or base64)
  - LLM 호출(선택): `POST {llm_url}/v1/completions`
  - 후처리(품질 보증): `remove_repetitions()`, `apply_table_fixes()`, `insert_chart_after_section2()`
  - 출력: `reports/report_YYYY_MM_{_llm_mcp|_prompt_mcp}_TIMESTAMP.md` (+ `reports/charts/*.png`)

### 5.2 Tools 서버 (HTTP)

- `report_agent/mcp_server/server.py`
  - FastAPI 기반
  - 주요 엔드포인트(일부):
    - `POST /tools/get_report_data`
    - `POST /tools/forecast_weekly_demand`
    - `POST /tools/generate_yearly_monthly_chart`
    - `GET /health`
  - 내부적으로 `CombinedTools()` / `ChartTools()`를 호출합니다.

### 5.3 Tools 서버 (Pure MCP)

- `report_agent/mcp_server/mcp_pure_server.py`
  - `FastMCP` 기반 “순수 MCP” 서버
  - 동일한 도구들을 `@mcp.tool()`로 노출합니다.
  - 실행 모드:
    - stdio: `python mcp_pure_server.py --mode stdio` (CLI가 자동 실행)
    - SSE: `python mcp_pure_server.py --mode sse --host 0.0.0.0 --port 8001` (선택)

### 5.4 Tools 구현 (DB/예측/차트)

- `report_agent/mcp_server/tools.py`
  - `DemandTools`: `demand_1hour` 기반 수요 요약/주차 집계/최대부하/과거동월
  - `WeatherTools`: `weather_1hour` 기반 월간 요약/과거동월 기상
  - `ForecastTools`: ARIMA/HW/LSTM/Ensemble 예측 + 월별 예측(`monthly_forecast`)
    - 주차별 LSTM 모델/스케일러 로드:
      - 4주: `artifacts/models/best_direct_lstm_full_weekly_max_h4_win12.pth` + `artifacts/scalers/scalers_weekly_max_h4_win12.pkl`
      - 8주: `artifacts/models/best_direct_lstm_full_weekly_max_h8_win12.pth` + `artifacts/scalers/scalers_weekly_max_h8_win12.pkl`
    - 월별 LSTM(평균/피크): `artifacts/models/best_many2one_lstm_state_monthly_*.pth` + `artifacts/scalers/scalers_monthly_*.pkl`
  - `ChartTools`: matplotlib로 PNG 생성 (HTTP는 파일 경로 반환, Pure MCP는 base64 반환 가능)
  - `CombinedTools`: 보고서에 필요한 데이터 패키징(`get_report_data`)

### 5.5 LLM 서버 (vLLM)

- `serve_vllm.py`
  - `power_demand_merged_model_llama3/`을 OpenAI 호환 API로 서빙합니다.
  - `generate_report.py`는 기본적으로 `POST /v1/completions`로 호출합니다.

---

## 6) MCP Tool 스펙 (입력/출력 “필드” 단위)

아래 스펙은 `report_agent/mcp_server/tools.py` 구현(및 `server.py` 노출) 기준입니다.

### 6.1 수요/기상 조회

| Tool | 입력 | 출력(핵심 필드) |
|---|---|---|
| `get_demand_summary` | `year`, `month` | `max_demand`, `avg_demand`, `min_demand`, `yoy_change` |
| `get_weekly_demand` | `year`, `month` | 각 주차 `max_demand/avg_demand/min_demand`, `date_range`, `start_date/end_date` |
| `get_peak_load` | `year`, `month` | `peak_datetime`, `peak_date`, `peak_hour`, `peak_demand`, `weekday` |
| `get_historical_demand` | `month`, `years`, `target_year?` | 연도별 `max_demand`, `avg_demand`, `max_yoy`, `avg_yoy` 등 |
| `get_weather_summary` (Pure MCP) | `year`, `month` | `temperature_avg/max/min`, `humidity_avg` |

> 참고: HTTP 모드에서는 `get_weather_summary`를 별도 엔드포인트로 제공하지 않고, `get_report_data` 안에 포함된 `weather`를 사용합니다.

### 6.2 예측

| Tool | 입력 | 출력(핵심 필드) |
|---|---|---|
| `forecast_weekly_demand` | `year`, `month`, `model`, `include_next_month` | `model`, `weeks_count`, `forecasts[]` (`week`, `date_range`, `max_demand`, `month/year`(다음달 포함 시)) |

### 6.3 차트

| Tool | 입력 | 출력(핵심 필드) |
|---|---|---|
| `generate_yearly_monthly_chart` | `target_year`, `target_month`, `years`, (`output_dir`는 HTTP만) | `success`, (HTTP: `filepath`), (Pure MCP: `image_base64`) |
| `generate_weekly_chart` | `year`, `month` | `success`, `filepath` 또는 `image_base64` |

---

## 7) 데이터/출력 포맷

### 7.1 SQLite 스키마(핵심 테이블)

- `demand_1hour`: `timestamp`, `demand_mean`, `demand_max`, `demand_min`, `is_holiday`, `day_type` 등
- `weather_1hour`: `timestamp`, `temperature_mean/max/min`, `humidity_mean/max/min`

### 7.2 출력 디렉토리

```
report_agent/reports/
├── report_2025_11_llm_mcp_YYYYMMDD_HHMMSS.md
└── charts/
    └── yearly_monthly_demand_2025_11.png
```

---

## 8) 의존성(필수 아티팩트/패키지)

### 8.1 필수 파일/폴더

- DB: `report_agent/demand_data/demand.db`
- vLLM 모델: `power_demand_merged_model_llama3/`
- 예측 모델/스케일러(아티팩트 폴더):
  - `artifacts/models/best_direct_lstm_full_weekly_max_h4_win12.pth`, `artifacts/scalers/scalers_weekly_max_h4_win12.pkl`
  - `artifacts/models/best_direct_lstm_full_weekly_max_h8_win12.pth`, `artifacts/scalers/scalers_weekly_max_h8_win12.pkl`
  - `artifacts/models/best_many2one_lstm_state_monthly_mean_win12.pth`, `artifacts/scalers/scalers_monthly_mean.pkl`
  - `artifacts/models/best_many2one_lstm_state_monthly_peak_win12.pth`, `artifacts/scalers/scalers_monthly_peak.pkl`

### 8.2 Python 패키지(주요)

- `mcp` (Pure MCP stdio/SSE)
- `fastapi`, `uvicorn` (HTTP 서버)
- `httpx` (HTTP 클라이언트)
- `torch` (LSTM 추론)
- `pandas`, `numpy` (데이터 처리)
- `statsmodels` (ARIMA, Holt-Winters)
- `matplotlib` (차트)
- `vllm` (LLM 서빙)

---

## 9) Troubleshooting (현상 → 원인/확인 포인트)

| 증상 | 확인 포인트 |
|---|---|
| MCP 서버 연결 실패(HTTP) | `curl http://localhost:8001/health`, 포트 충돌, `--mcp-url` |
| Pure MCP stdio 실패 | `mcp` 패키지 설치, `report_agent/mcp_server/mcp_pure_server.py` 실행 가능 여부 |
| vLLM 호출 실패 | `--llm-url` 정확도, 8000 포트, GPU 메모리, `power_demand_merged_model_llama3/` 경로 |
| 예측 실패/빈 값 | `demand.db`에 과거 데이터 충분한지, `artifacts/models`/`artifacts/scalers` 존재 여부 |
| 차트 생성 실패 | `matplotlib` 설치, `report_agent/reports/` 쓰기 권한, 데이터 유무 |
