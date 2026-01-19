"""
LoRA 어댑터를 베이스 모델에 머지하는 스크립트
머지된 모델은 vLLM으로 서빙 가능
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# 경로 설정 - llama -> 이 부분만 수정하면 됨.
BASE_MODEL_PATH = "/root/models/llama-3-korean-bllossom-8B"
ADAPTER_PATH = "/root/De-Qwen-SFT/power_demand_sft_model_llama3"
OUTPUT_PATH = "./power_demand_merged_model_llama3"

def merge_lora():
    print("=" * 60)
    print("LoRA 어댑터 머지 시작")
    print("=" * 60)

    # 1. 토크나이저 로딩
    print("\n[1/4] 토크나이저 로딩...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    # 2. 베이스 모델 로딩 (FP16, 양자화 없이)
    print("\n[2/4] 베이스 모델 로딩 (FP16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    # 3. LoRA 어댑터 적용
    print("\n[3/4] LoRA 어댑터 적용...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 4. 머지 및 언로드
    print("\n[4/4] 머지 및 저장...")
    merged_model = model.merge_and_unload()

    # 저장
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"\n머지 완료!")
    print(f"저장 경로: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    merge_lora()
