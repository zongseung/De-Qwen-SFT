"""
전력수요 문서에서 GPT API를 활용한 다양한 형태의 학습 데이터 생성
- 짧은 Q&A
- 긴 분석/설명
- 요약
- 보고서 작성
"""

import os
import glob
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import time
import traceback

# OpenAI API 키 설정
api_key = os.getenv("OPENAI_API_KEY", "")
# 특수 문자 제거 (모든 종류의 따옴표와 제어 문자)
if api_key:
    for quote_char in ['"', '"', '"', "'", '\u201c', '\u201d', '\u2018', '\u2019']:
        api_key = api_key.replace(quote_char, '')
    api_key = api_key.strip()
    api_key = api_key.encode('ascii', errors='ignore').decode('ascii')

client = OpenAI(api_key=api_key)

# 다양한 Task 생성 프롬프트
DIVERSE_TASK_PROMPT = """
당신은 전력수요 예측 전문가입니다. 주어진 문서에서 **5-8개의 다양한 형태의 학습 데이터**를 생성해주세요.

## 데이터 유형 (반드시 다양하게 섞을 것):

### 1. 짧은 Q&A (30% 비율)
- 간단한 사실 확인 질문
- 답변: 1-2문장

### 2. 분석/설명 (30% 비율)
- 추세 분석, 비교, 원인 설명 등
- 답변: 3-5문장의 구조화된 설명

### 3. 요약 (20% 비율)
- 문서 전체 또는 일부 섹션 요약
- 답변: 3-4문장의 핵심 요약

### 4. 보고서 작성 (20% 비율)
- 전문적인 보고서 형태
- 답변: 구조화된 긴 형태 (제목, 본문, 결론)

## 출력 형식:
```json
[
  {{
    "type": "short_qa",
    "question": "2019년 2월 최대부하는?",
    "answer": "8,520만kW입니다."
  }},
  {{
    "type": "analysis",
    "question": "2018년과 2019년 2월 전력수요를 비교 분석하세요.",
    "answer": "2018년 2월 최대부하는 8,888만kW였으나, 2019년은 8,520만kW로 4.1% 감소했습니다. 이는 기온 변화와 에너지 효율 개선이 주요 원인으로 분석됩니다. 평균부하 역시 0.5% 감소하여 전반적인 수요 감소 추세를 보였습니다."
  }},
  {{
    "type": "summary",
    "question": "이 문서의 핵심 내용을 요약하세요.",
    "answer": "2019년 2월 전력수요는 최대부하 8,520만kW, 평균부하 6,840만kW로 예측되었습니다. 전년 대비 각각 4.1%, 0.5% 감소했으며, 기온은 평년과 비슷하나 변화폭이 클 것으로 전망되었습니다. 강수량은 전반에 많고 후반에는 평년 수준일 것으로 예상되었습니다."
  }},
  {{
    "type": "report",
    "question": "2019년 2월 전력수요 분석 보고서를 작성하세요.",
    "answer": "# 2019년 2월 전력수요 분석 보고서\\n\\n## 1. 개요\\n2019년 2월 전력수요는 최대부하 8,520만kW로 전년 대비 4.1% 감소가 예상됩니다.\\n\\n## 2. 주요 지표\\n- 최대부하: 8,520만kW (-4.1%)\\n- 평균부하: 6,840만kW (-0.5%)\\n- 예측 시점: 2019년 1월\\n\\n## 3. 기상 조건\\n기온은 평년과 비슷하나 변동성이 크며, 강수량은 전반기에 많을 것으로 전망됩니다.\\n\\n## 4. 결론\\n전년 대비 감소세가 뚜렷하며, 기온 변화에 따른 변동 가능성을 고려한 운영이 필요합니다."
  }}
]
```

## 중요:
- 한글과 영어를 섞어서 생성하되, 긴 답변은 한글로
- type을 명시적으로 표시
- 답변 길이를 다양하게 (짧은 것부터 긴 것까지)
- JSON 형태로만 출력 (다른 텍스트 불포함)

## 문서:
{document}
"""


def load_all_markdown_files(base_dir="."):
    """모든 마크다운 파일 로딩"""
    md_files = []

    for year in ["2019", "2020"]:
        pattern = f"{base_dir}/{year}_md/*.md"
        md_files.extend(glob.glob(pattern))

    for year in ["2021", "2022", "2023", "2024"]:
        pattern = f"{base_dir}/{year}_md/**/*.md"
        md_files.extend(glob.glob(pattern, recursive=True))

    print(f"총 {len(md_files)}개의 마크다운 파일 발견")
    return md_files


def read_file_content(file_path):
    """파일 내용 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if len(content) > 10000:
                content = content[:10000] + "\n\n[문서가 너무 길어 일부만 포함됨]"
            return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def generate_diverse_tasks(document_content, model="gpt-4o-mini", max_retries=5):
    """
    GPT API를 사용하여 문서에서 다양한 형태의 학습 데이터 생성
    Rate limit 방어 로직 포함
    """
    prompt = DIVERSE_TASK_PROMPT.format(document=document_content)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "당신은 전력수요 예측 데이터에서 다양한 형태의 학습 데이터를 생성하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000,  # 긴 답변을 위해 증가
            )

            response_text = response.choices[0].message.content.strip()

            # JSON 코드 블록 제거
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # JSON 파싱
            tasks = json.loads(response_text)

            return tasks

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 에러 (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"응답 내용: {response_text[:200]}...")
            if attempt == max_retries - 1:
                return []
            time.sleep(2)

        except Exception as e:
            error_str = str(e)

            # Rate limit 에러 감지
            if "rate_limit" in error_str.lower() or "429" in error_str:
                wait_time = min(60, (2 ** attempt) * 5)  # Exponential backoff (최대 60초)
                print(f"⚠️  Rate limit 도달! {wait_time}초 대기 중... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"API 호출 에러 (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == 0:
                    print("Full traceback:")
                    traceback.print_exc()
                time.sleep(3)

            if attempt == max_retries - 1:
                return []

    return []


def process_all_documents(base_dir=".", output_file="diverse_qa_dataset.json", limit=None, delay=5.0):
    """
    모든 문서 처리하여 다양한 형태의 학습 데이터 생성

    Args:
        base_dir: 문서 디렉토리
        output_file: 출력 파일명
        limit: 처리할 파일 수 제한
        delay: API 호출 간 대기 시간 (초) - rate limit 방지
    """
    md_files = load_all_markdown_files(base_dir)

    if limit:
        md_files = md_files[:limit]
        print(f"처리 제한: {limit}개 파일만 처리")

    # 체크포인트 파일 확인 (이전 실행 중단 시 재개용)
    checkpoint_file = Path(base_dir) / f"{output_file}.checkpoint"
    processed_files = set()
    all_tasks = []

    if checkpoint_file.exists():
        print(f"✓ 체크포인트 발견: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            all_tasks = checkpoint_data.get('tasks', [])
            processed_files = set(checkpoint_data.get('processed_files', []))
        print(f"  - 이미 처리된 파일: {len(processed_files)}개")
        print(f"  - 이미 생성된 Task: {len(all_tasks)}개")
        print(f"  - 남은 파일: {len([f for f in md_files if f not in processed_files])}개\n")

    failed_files = []

    print(f"\n{len(md_files)}개 문서 처리 시작...")
    print(f"⏱️  API 호출 간격: {delay}초 (rate limit 방지)\n")

    for i, file_path in enumerate(tqdm(md_files, desc="다양한 Task 생성 중")):
        # 이미 처리한 파일 스킵
        if file_path in processed_files:
            continue

        content = read_file_content(file_path)
        if not content or len(content.strip()) < 100:
            print(f"스킵: {file_path} (내용 부족)")
            processed_files.add(file_path)
            continue

        tasks = generate_diverse_tasks(content)

        if tasks:
            for task in tasks:
                task['source_file'] = file_path
            all_tasks.extend(tasks)
            print(f"✓ {file_path}: {len(tasks)}개 생성")
        else:
            failed_files.append(file_path)
            print(f"✗ {file_path}: 생성 실패")

        processed_files.add(file_path)

        # 10개 파일마다 체크포인트 저장
        if (i + 1) % 10 == 0:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'tasks': all_tasks,
                    'processed_files': list(processed_files)
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 체크포인트 저장: {len(all_tasks)}개 Task")

        # Rate limit 방지를 위한 대기
        if i < len(md_files) - 1:  # 마지막 파일이 아니면
            time.sleep(delay)

    # 최종 결과 저장
    output_path = Path(base_dir) / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_tasks, f, ensure_ascii=False, indent=2)

    # 체크포인트 파일 삭제 (완료 후)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"✓ 체크포인트 파일 삭제")

    # 통계 출력
    task_types = {}
    for task in all_tasks:
        task_type = task.get('type', 'unknown')
        task_types[task_type] = task_types.get(task_type, 0) + 1

    print(f"\n{'='*60}")
    print(f"✓ 총 {len(all_tasks)}개 학습 데이터 생성 완료")
    print(f"✓ 저장 위치: {output_path}")
    print(f"\n📊 Task 유형별 분포:")
    for task_type, count in task_types.items():
        print(f"  - {task_type}: {count}개")
    print(f"\n✗ 실패한 파일: {len(failed_files)}개")

    if failed_files:
        print(f"\n실패한 파일 목록:")
        for f in failed_files[:10]:
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... 외 {len(failed_files) - 10}개")

    return all_tasks


def convert_to_sft_format(dataset_file, output_file="diverse_sft_dataset.jsonl"):
    """
    다양한 형태의 데이터를 SFT용 포맷으로 변환
    """
    with open(dataset_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    sft_data = []
    for task in tasks:
        sft_data.append({
            "messages": [
                {"role": "user", "content": task['question']},
                {"role": "assistant", "content": task['answer']}
            ],
            "type": task.get('type', 'unknown')
        })

    # JSONL 형태로 저장
    output_path = Path(dataset_file).parent / output_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✓ SFT 포맷 데이터셋 생성 완료: {output_path}")
    print(f"✓ 총 {len(sft_data)}개 대화 쌍")

    return sft_data


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        exit(1)

    # 1. 다양한 형태의 학습 데이터 생성
    tasks = process_all_documents(
        base_dir=".",
        output_file="diverse_qa_dataset.json",
        limit=5,  # 테스트: limit=5, 전체: limit=None
        delay=5.0  # API 호출 간격 (초) - rate limit 방지
    )

    # 2. SFT 포맷으로 변환
    if tasks:
        convert_to_sft_format("diverse_qa_dataset.json", "diverse_sft_dataset.jsonl")

        # 샘플 출력 (각 타입별)
        print("\n=== 샘플 데이터 (타입별) ===")
        task_types_shown = set()
        for i, task in enumerate(tasks):
            task_type = task.get('type', 'unknown')
            if task_type not in task_types_shown:
                print(f"\n[{task_type.upper()}]")
                print(f"질문: {task['question']}")
                print(f"답변: {task['answer'][:200]}...")
                task_types_shown.add(task_type)
                if len(task_types_shown) >= 4:
                    break
