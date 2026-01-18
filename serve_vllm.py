"""
vLLM을 사용한 전력수요 특화 모델 서빙
OpenAI 호환 API 서버로 실행됨
"""

from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
import argparse

# 모델 경로
MODEL_PATH = "./power_demand_merged_model"

def create_llm():
    """vLLM 엔진 생성"""
    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.7,
        max_num_seqs=32,
    )
    return llm

def generate_response(llm, prompt: str, max_tokens: int = 1024):
    """단일 응답 생성"""
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Qwen 챗 템플릿 적용
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    outputs = llm.generate([formatted_prompt], sampling_params)
    return outputs[0].outputs[0].text

def interactive_mode(llm):
    """대화형 테스트 모드"""
    print("\n" + "=" * 60)
    print("전력수요 특화 모델 (vLLM)")
    print("종료하려면 'quit' 또는 'exit' 입력")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n[You] ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("종료합니다.")
                break
            if not prompt.strip():
                continue

            response = generate_response(llm, prompt)
            print(f"\n[AI] {response}")

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

def main():
    parser = argparse.ArgumentParser(description="전력수요 특화 모델 vLLM 서빙")
    parser.add_argument("--mode", choices=["interactive", "server"], default="interactive",
                        help="실행 모드 (interactive: 대화형, server: API 서버)")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트")
    args = parser.parse_args()

    if args.mode == "interactive":
        print("모델 로딩 중...")
        llm = create_llm()
        interactive_mode(llm)
    else:
        # OpenAI 호환 API 서버 실행
        print(f"API 서버 시작: http://{args.host}:{args.port}")
        print("OpenAI 호환 엔드포인트:")
        print(f"  - POST /v1/chat/completions")
        print(f"  - POST /v1/completions")

        # vLLM CLI로 서버 실행하는 것이 더 안정적
        import subprocess
        subprocess.run([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_PATH,
            "--host", args.host,
            "--port", str(args.port),
            "--dtype", "float16",
            "--trust-remote-code",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.8",
        ])

if __name__ == "__main__":
    main()
