"""
SFT 학습된 전력수요 특화 모델 테스트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from report_prompt_template import generate_monthly_report_prompt, generate_simple_report_prompt

print("=" * 80)
print("전력수요 특화 모델 테스트")
print("=" * 80)

# 1. 모델 로딩
print("\n[1] 모델 로딩 중...")

# 베이스 모델 경로
base_model_path = "./model_weights"
# SFT adapter 경로
adapter_path = "./power_demand_sft_model"

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 베이스 모델 로딩
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# LoRA adapter 적용
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print(f"✓ 모델 로딩 완료!")
print(f"  Device: {model.device}")


# 2. 생성 함수
def generate_response(prompt, max_new_tokens=500):
    """모델 응답 생성"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
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
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response


# 3. 테스트 실행
print("\n" + "=" * 80)
print("[2] Q&A 테스트")
print("=" * 80)

qa_tests = [
    "2019년 1월 최대부하 예측은 얼마인가요?",
    "여름철 전력수요가 높은 이유를 설명해주세요.",
    "전력수요 예측에서 기온의 영향은 어떻게 되나요?",
    "What is the typical peak electricity demand in August?",
]

for i, question in enumerate(qa_tests, 1):
    print(f"\n[Q{i}] {question}")
    answer = generate_response(question, max_new_tokens=200)
    print(f"[A{i}] {answer}")
    print("-" * 80)


# 4. 보고서 생성 테스트
print("\n" + "=" * 80)
print("[3] 보고서 생성 테스트")
print("=" * 80)

# 테스트 데이터 (2025년 1월 - 임의 데이터)
test_data = {
    "year": 2025,
    "month": 1,
    "max_load": 9150,
    "avg_load": 7380,
    "yoy_max_change": "+2.8",
    "yoy_avg_change": "+1.5",
    "temp_forecast": "대체로 평년과 비슷하거나 높겠으나, 기온 변동성이 크겠으며 대륙고기압의 영향으로 기온이 큰 폭으로 떨어질 때가 있겠음",
    "precip_forecast": "대체로 평년과 비슷하겠음",
    "historical_data": """
- 2021년 1월: 최대 8,520만kW (+1.2%), 평균 6,980만kW (+0.8%)
- 2022년 1월: 최대 8,650만kW (+1.5%), 평균 7,050만kW (+1.0%)
- 2023년 1월: 최대 8,780만kW (+1.5%), 평균 7,150만kW (+1.4%)
- 2024년 1월: 최대 8,900만kW (+1.4%), 평균 7,270만kW (+1.7%)
- 2025년 1월: 최대 9,150만kW (+2.8%), 평균 7,380만kW (+1.5%) [예측]
""",
    "weekly_data": """
- 1주차 (12/30~1/5): 8,850만kW
- 2주차 (1/6~1/12): 9,050만kW
- 3주차 (1/13~1/19): 9,150만kW (월 최대)
- 4주차 (1/20~1/26): 9,100만kW
- 5주차 (1/27~2/2): 8,950만kW
""",
    "methodology": """
○ ARIMA, 지수평활, 월별 회귀분석을 활용한 결과값의 평균값으로 예측
- Holt-Winters 지수평활 예측: 7,425만kW
- 월별 수요 변동특성 및 회귀분석 예측: 7,320만kW
- 월간예측모형(SEFS) 예측 결과: 7,395만kW
"""
}

# 상세 보고서 프롬프트
report_prompt = generate_monthly_report_prompt(test_data)

print("\n[프롬프트]")
print(report_prompt[:500] + "...")

print("\n[생성된 보고서]")
report = generate_response(report_prompt, max_new_tokens=1000)
print(report)


# 5. 간단 보고서 테스트
print("\n" + "=" * 80)
print("[4] 간단 보고서 테스트")
print("=" * 80)

simple_data = {
    "year": 2025,
    "month": 1,
    "max_load": 9150,
    "avg_load": 7380,
    "yoy_change": "+2.8",
    "temp_forecast": "평년과 비슷하거나 높음, 기온 변동성 큼"
}

simple_prompt = generate_simple_report_prompt(simple_data)
print("\n[프롬프트]")
print(simple_prompt)

print("\n[생성된 보고서]")
simple_report = generate_response(simple_prompt, max_new_tokens=500)
print(simple_report)

# 6. 보고서 마크다운 파일로 저장
print("\n" + "=" * 80)
print("[5] 보고서 저장")
print("=" * 80)

import os
from datetime import datetime

# 저장 디렉토리 생성
output_dir = "./generated_reports"
os.makedirs(output_dir, exist_ok=True)

# 파일명 생성 (날짜 + 대상월)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = f"{output_dir}/power_demand_report_{test_data['year']}_{test_data['month']:02d}_{timestamp}.md"

# 보고서 저장
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"✓ 보고서 저장 완료: {report_filename}")

# 간단 보고서도 저장
simple_filename = f"{output_dir}/power_demand_simple_{test_data['year']}_{test_data['month']:02d}_{timestamp}.md"
with open(simple_filename, 'w', encoding='utf-8') as f:
    f.write(simple_report)

print(f"✓ 간단 보고서 저장 완료: {simple_filename}")

print("\n" + "=" * 80)
print("✓ 테스트 완료!")
print("=" * 80)

