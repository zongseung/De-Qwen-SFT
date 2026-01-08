"""
자동 예측 + 보고서 생성 테스트
- 과거 데이터만 주고 예측값과 보고서를 모두 생성하게 함
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("=" * 80)
print("자동 예측 + 보고서 생성 테스트")
print("=" * 80)

# 1. 모델 로딩
print("\n[1] 모델 로딩 중...")

base_model_path = "./model_weights"
adapter_path = "./power_demand_sft_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("✓ 모델 로딩 완료!")


def generate_response(prompt, max_new_tokens=1500):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# 2. 자동 예측 프롬프트 (과거 데이터만 제공)
AUTO_FORECAST_PROMPT = """
당신은 한국전력거래소의 전력수요 예측 전문가입니다.
과거 데이터를 분석하여 2025년 1월 전력수요를 예측하고, 보고서를 작성해주세요.

## 과거 전력수요 실적 (최근 5개년 1월)

| 연도 | 최대부하(만kW) | 전년대비 | 평균부하(만kW) | 전년대비 |
|------|---------------|---------|---------------|---------|
| 2020년 1월 | 8,350 | +2.1% | 6,820 | +1.8% |
| 2021년 1월 | 8,520 | +2.0% | 6,980 | +2.3% |
| 2022년 1월 | 8,650 | +1.5% | 7,050 | +1.0% |
| 2023년 1월 | 8,780 | +1.5% | 7,150 | +1.4% |
| 2024년 1월 | 8,900 | +1.4% | 7,270 | +1.7% |

## 2025년 1월 기상 전망
- 기온: 평년과 비슷하거나 높겠으나, 대륙고기압 영향으로 기온 변동성이 크겠음
- 강수량: 평년과 비슷하겠음

## 요청사항
1. 위 데이터를 분석하여 2025년 1월 최대부하와 평균부하를 예측해주세요.
2. 예측 결과를 바탕으로 전력수요 예측 보고서를 작성해주세요.

## 보고서 형식

# 2025년 01월 전력수요 예측 전망

## 1. 기상전망
(기온 및 강수량 전망)

## 2. 과거 전력수요 추이
(최근 5개년 실적 분석)

## 3. 전력수요 전망 결과
- 최대부하 예측: ____만kW (전년 대비 ___%)
- 평균부하 예측: ____만kW (전년 대비 ___%)

### 주별 최대전력 예측
(주차별 예측값)

**방법론/산식**
(예측 방법 설명)

## 4. 종합 분석
(주요 변동 요인 및 시사점)
"""


print("\n" + "=" * 80)
print("[2] 자동 예측 + 보고서 생성")
print("=" * 80)

print("\n[과거 데이터만 제공하고 예측 + 보고서 작성 요청]")
report = generate_response(AUTO_FORECAST_PROMPT, max_new_tokens=1500)

print("\n" + "-" * 80)
print("[생성된 보고서]")
print("-" * 80)
print(report)

# 저장
import os
from datetime import datetime

output_dir = "./generated_reports"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{output_dir}/auto_forecast_report_2025_01_{timestamp}.md"

with open(filename, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ 보고서 저장: {filename}")
print("=" * 80)

