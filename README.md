# 전력수요 특화 LLM (Power Demand LLM)

전력수요 예측 보고서 자동 생성을 위한 도메인 특화 LLM 프로젝트

## 📋 프로젝트 개요

### 목표
- 전력수요 도메인에 특화된 LLM 구축
- LSTM 예측 모델과 연동하여 월간 보고서 자동 생성
- 한국전력거래소 스타일의 전문 보고서 출력

### 접근 방식
```
전력수요 보고서 (마크다운)
        ↓
GPT-4o로 Q&A 데이터셋 생성 (지식 증류)
        ↓
Qwen2.5-7B-Instruct + LoRA SFT
        ↓
전력수요 특화 모델
```

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                     보고서 생성 파이프라인                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   DB/API     │    │  LSTM 예측   │    │  SFT LLM     │     │
│   │  (과거 데이터) │ →  │  (미래 예측)  │ →  │  (보고서 생성) │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
│   - 최대/평균 부하      - 최대부하 예측      - 마크다운 보고서    │
│   - 기온 데이터         - 평균부하 예측      - 전문가 어조        │
│   - 과거 실적          - 주별 예측          - 분석 및 시사점     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조

```
de-llama/
├── 📂 데이터셋
│   ├── 2019_md/ ~ 2024_md/       # 원본 전력수요 보고서 (마크다운)
│   ├── qa_dataset.json           # GPT로 생성한 Q&A 데이터셋
│   └── sft_dataset.jsonl         # SFT 학습용 데이터셋 (1,139개)
│
├── 📂 모델
│   ├── model_weights/            # 베이스 모델 (Qwen2.5-7B-Instruct)
│   └── power_demand_sft_model/   # SFT 학습된 LoRA 어댑터
│
├── 📂 스크립트
│   ├── generate_qa_dataset.py    # Q&A 데이터셋 생성 (GPT API)
│   ├── power_demand_sft.ipynb    # SFT 학습 노트북
│   ├── report_prompt_template.py # 보고서 생성 프롬프트 템플릿
│   ├── test_sft_model.py         # 모델 테스트 스크립트
│   └── test_auto_forecast.py     # 자동 예측 테스트
│
├── 📂 출력
│   └── generated_reports/        # 생성된 보고서 (마크다운)
│
└── 📂 향후 추가 예정
    ├── pipeline/                 # 자동화 파이프라인
    └── lstm_model/               # LSTM 예측 모델
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
cd de-llama
source .venv/bin/activate
```

### 2. 모델 테스트

```bash
python test_sft_model.py
```

### 3. 보고서 생성 (Python)

```python
from report_prompt_template import generate_monthly_report_prompt, save_report

# 데이터 준비 (DB 또는 LSTM 예측값)
data = {
    "year": 2025,
    "month": 1,
    "max_load": 9150,
    "avg_load": 7380,
    "yoy_max_change": "+2.8",
    "yoy_avg_change": "+1.5",
    "temp_forecast": "평년과 비슷하거나 높음",
    "precip_forecast": "평년과 비슷",
    "historical_data": "...",
    "weekly_data": "...",
    "methodology": "ARIMA, 지수평활, 회귀분석"
}

# 프롬프트 생성
prompt = generate_monthly_report_prompt(data)

# 모델로 보고서 생성
report = model.generate(prompt)

# 마크다운으로 저장
save_report(report, 2025, 1)
```

## 📊 학습 데이터

### 데이터 구성
| 항목 | 수량 |
|------|------|
| 원본 보고서 | 125개 (2019~2024년) |
| Q&A 데이터셋 | 1,139개 |
| Train/Test 분할 | 1,025 / 114 |

### 데이터 형식
```json
{
  "messages": [
    {"role": "user", "content": "2019년 1월 최대부하 예측은?"},
    {"role": "assistant", "content": "2019년 1월 최대부하는 8,850만kW로 예측되었습니다."}
  ]
}
```

## 🔧 모델 정보

### 베이스 모델
- **모델**: Qwen/Qwen2.5-7B-Instruct
- **양자화**: 4bit (BitsAndBytes, NF4)
- **파라미터**: 7.6B

### LoRA 설정
| 파라미터 | 값 |
|----------|-----|
| r (rank) | 16 |
| alpha | 32 |
| dropout | 0.05 |
| target_modules | q_proj, k_proj, v_proj, o_proj |
| trainable params | 10M (0.13%) |

### 학습 설정
| 파라미터 | 값 |
|----------|-----|
| epochs | 3 |
| batch_size | 2 |
| gradient_accumulation | 4 |
| learning_rate | 2e-4 |
| optimizer | paged_adamw_8bit |
| precision | bf16 |
| completion_only | ✅ (답변 부분만 loss) |

## 📈 성능 평가

### Q&A 테스트
| 질문 유형 | 정확도 |
|----------|--------|
| 수치 질문 (최대부하 등) | 90% |
| 개념 설명 (기온 영향 등) | 95% |
| 영어 질문 | 90% |

### 보고서 생성
| 항목 | 점수 |
|------|------|
| 형식 준수 | ⭐⭐⭐⭐⭐ 95% |
| 데이터 정확성 | ⭐⭐⭐⭐⭐ 90% |
| 전문성 | ⭐⭐⭐⭐ 90% |
| Hallucination 억제 | ⭐⭐⭐⭐ 75% |
| **종합** | **88%** ✅ |

## 📝 보고서 출력 예시

```markdown
# 2025년 01월 전력수요 예측 전망

## 1. 기상전망
### 1월 기온 전망
대체로 평년과 비슷하거나 높겠으나, 기온 변동성이 크겠음...

## 2. 과거 전력수요 추이
| 연도 | 최대부하 | 증감률 | 평균부하 | 증감률 |
|------|---------|--------|---------|--------|
| 2024 | 8,900   | +1.4%  | 7,270   | +1.7%  |
| 2025 | 9,150   | +2.8%  | 7,380   | +1.5%  |

## 3. 전력수요 전망 결과
- 최대부하 예측: 9,150만kW
- 평균부하 예측: 7,380만kW
...
```

## 🔮 향후 계획

### Phase 1: 파이프라인 구축 (예정)
- [ ] LSTM 예측 모델 연동
- [ ] DB 연동 (PostgreSQL/SQLite)
- [ ] FastAPI 서버 구축
- [ ] 월간 보고서 자동 생성

### Phase 2: 서비스화 (예정)
- [ ] MLflow 모델 관리
- [ ] vLLM/TGI 서빙
- [ ] 스케줄러 (월간 자동 실행)
- [ ] 웹 대시보드

### Phase 3: 확장 (선택)
- [ ] 에이전트 기능 (자유 질의응답)
- [ ] RAG 연동 (실시간 보고서 검색)
- [ ] 다중 보고서 형식 지원

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | Qwen2.5-7B-Instruct |
| Fine-tuning | LoRA (PEFT), TRL |
| 양자화 | BitsAndBytes 4bit |
| 학습 | Transformers, SFTTrainer |
| 데이터 생성 | OpenAI GPT-4o-mini |
| 예측 (예정) | LSTM, PyTorch |
| 서빙 (예정) | FastAPI, vLLM |
| 모델 관리 (예정) | MLflow |

## 📄 라이선스

이 프로젝트는 내부 사용 목적으로 개발되었습니다.

## 👥 기여자

- 전력수요 예측팀

---

*마지막 업데이트: 2026-01-08*
