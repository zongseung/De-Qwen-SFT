#!/usr/bin/env python
"""
전력수요 보고서 생성 CLI

Usage:
    # 프롬프트만 생성 (LLM 없이)
    python generate_report.py --year 2025 --month 12

    # LLM 서버로 보고서 생성
    python generate_report.py --year 2025 --month 12 --llm-url http://localhost:8000

    # 출력 디렉토리 지정
    python generate_report.py --year 2025 --month 12 --output ./reports
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (시스템에 따라 조정 필요)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.tools import CombinedTools


def generate_weekly_chart(
    weekly_data: List[Dict[str, Any]],
    year: int,
    month: int,
    output_dir: Path,
    historical_data: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """주별 전력수요 추이 라인 차트 생성 (다년간 비교 포함)
    
    Args:
        weekly_data: 주별 전력수요 데이터
        year: 연도
        month: 월
        output_dir: 출력 디렉토리
        historical_data: 과거 N년간 동월 데이터 (선택)
        
    Returns:
        생성된 차트 파일 경로
    """
    if not weekly_data:
        return None
    
    # 차트 저장 디렉토리 생성
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 추출
    weeks = [f"W{d['week']}" for d in weekly_data]
    max_demands = [d['max_demand'] / 10000 if d['max_demand'] else 0 for d in weekly_data]
    avg_demands = [d['avg_demand'] / 10000 if d['avg_demand'] else 0 for d in weekly_data]
    
    # 차트 생성 (다년간 비교가 있으면 2개 subplot)
    if historical_data and len(historical_data) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 왼쪽: 현재 연도 주별 수요 차트
        ax1.plot(weeks, max_demands, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Max')
        ax1.plot(weeks, avg_demands, 's-', color='#3498db', linewidth=2, markersize=8, label='Avg')
        ax1.fill_between(weeks, [0]*len(weeks), max_demands, alpha=0.1, color='#e74c3c')
        ax1.set_xlabel('Week', fontsize=11)
        ax1.set_ylabel('Power Demand (10MW)', fontsize=11)
        ax1.set_title(f'{year}/{month:02d} Weekly Demand', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(bottom=0)
        ax1.set_facecolor('#f8f9fa')
        
        # 오른쪽: 다년간 동월 비교 막대 차트
        hist_years = [str(h['year']) for h in historical_data]
        hist_max = [h['max_demand'] / 10000 if h['max_demand'] else 0 for h in historical_data]
        hist_avg = [h['avg_demand'] / 10000 if h['avg_demand'] else 0 for h in historical_data]
        
        x = range(len(hist_years))
        width = 0.35
        
        bars1 = ax2.bar([i - width/2 for i in x], hist_max, width, label='Max', color='#e74c3c', alpha=0.8)
        bars2 = ax2.bar([i + width/2 for i in x], hist_avg, width, label='Avg', color='#3498db', alpha=0.8)
        
        # 트렌드 라인 추가
        ax2.plot(x, hist_max, 'o--', color='#c0392b', linewidth=1.5, markersize=4)
        ax2.plot(x, hist_avg, 'o--', color='#2980b9', linewidth=1.5, markersize=4)
        
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Power Demand (10MW)', fontsize=11)
        ax2.set_title(f'Month {month} Historical Comparison ({hist_years[-1]}-{hist_years[0]})', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(hist_years, rotation=45, ha='right')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax2.set_ylim(bottom=0)
        ax2.set_facecolor('#f8f9fa')
        
        # 현재 연도 하이라이트
        current_idx = next((i for i, h in enumerate(historical_data) if h['year'] == year), None)
        if current_idx is not None:
            bars1[current_idx].set_edgecolor('black')
            bars1[current_idx].set_linewidth(2)
            bars2[current_idx].set_edgecolor('black')
            bars2[current_idx].set_linewidth(2)
        
        fig.patch.set_facecolor('white')
    else:
        # 히스토리컬 데이터 없으면 단일 차트
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(weeks, max_demands, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Max (10MW)')
        ax.plot(weeks, avg_demands, 's-', color='#3498db', linewidth=2, markersize=8, label='Avg (10MW)')
        ax.fill_between(weeks, [0]*len(weeks), max_demands, alpha=0.1, color='#e74c3c')
        
        for w, max_d, avg_d in zip(weeks, max_demands, avg_demands):
            ax.annotate(f'{max_d:.1f}', (w, max_d), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9, color='#e74c3c')
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Power Demand (10MW)', fontsize=12)
        ax.set_title(f'Weekly Power Demand Trend - {year}/{month:02d}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
    
    # 파일 저장
    filename = f"weekly_demand_{year}_{month:02d}.png"
    filepath = charts_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def build_prompt(data: Dict[str, Any]) -> str:
    """보고서 생성용 프롬프트 구성"""
    summary = data.get("summary", {})
    weather = data.get("weather", {})
    weekly = data.get("weekly_demand", [])
    historical = data.get("historical", [])
    peak = data.get("peak_load", {})

    year = summary.get("year", 0)
    month = summary.get("month", 0)

    # 과거 데이터 문자열
    hist_lines = []
    for h in historical:
        if h.get("max_demand"):
            hist_lines.append(
                f"- {h['year']}년: 최대 {h['max_demand']/10000:.1f}만kW, "
                f"평균 {h['avg_demand']/10000:.1f}만kW"
            )
    hist_str = "\n".join(hist_lines) if hist_lines else "데이터 없음"

    # 주별 데이터 문자열
    weekly_lines = []
    for w in weekly:
        if w.get("max_demand"):
            weekly_lines.append(
                f"- {w['week']}주차: 최대 {w['max_demand']/10000:.1f}만kW, "
                f"평균 {w['avg_demand']/10000:.1f}만kW"
            )
    weekly_str = "\n".join(weekly_lines) if weekly_lines else "데이터 없음"

    # 기상 정보
    temp_info = ""
    if weather and not weather.get("error"):
        temp_info = f"""- 평균기온: {weather.get('temperature_avg', 0):.1f}°C
- 최고기온: {weather.get('temperature_max', 0):.1f}°C
- 최저기온: {weather.get('temperature_min', 0):.1f}°C
- 평균습도: {weather.get('humidity_avg', 0):.1f}%"""
    else:
        temp_info = "기상 데이터 없음"

    # 최대부하 정보
    peak_info = ""
    if peak and not peak.get("error"):
        peak_info = f"- 최대부하 발생: {peak.get('peak_date', '')} {peak.get('peak_hour', 0)}시 ({peak.get('weekday', '')})"

    # YoY 변화
    yoy = summary.get("yoy_change")
    yoy_str = f"{yoy:+.1f}%" if yoy else "N/A"

    prompt = f"""당신은 한국전력거래소의 전력수요 예측 전문가입니다.
다음 데이터를 기반으로 {year}년 {month}월 전력수요 분석 보고서를 작성해주세요.

## 입력 데이터

### 기본 정보
- 대상 기간: {year}년 {month}월
- 최대부하: {summary.get('max_demand', 0)/10000:.1f}만kW (전년 동월 대비 {yoy_str})
- 평균부하: {summary.get('avg_demand', 0)/10000:.1f}만kW
- 최소부하: {summary.get('min_demand', 0)/10000:.1f}만kW
{peak_info}

### 기상 현황
{temp_info}

### 주별 전력수요
{weekly_str}

### 과거 동월 실적 (최근 5개년)
{hist_str}

## 보고서 작성 지침
1. 전문적이고 객관적인 어조로 작성
2. 수치는 만kW 단위로 표기
3. 전년 대비 증감 분석 포함
4. 기상 영향 분석 포함 (데이터 있는 경우)

## 보고서 형식

# {year}년 {month}월 전력수요 분석 보고서

## 1. 개요
(해당 월 전력수요 요약)

## 2. 전력수요 현황
(최대/평균/최소 부하 분석, 주별 추이)

## 3. 기상 영향 분석
(기온과 전력수요 상관관계)

## 4. 전년 동월 대비 분석
(YoY 변화 원인 분석)

## 5. 결론
(종합 분석 및 시사점)
"""
    return prompt


def remove_repetitions(text: str) -> str:
    """연속으로 반복되는 문장/단락 제거 및 불필요한 텍스트 정리"""
    import re
    
    if not text or not text.strip():
        return text
    
    # 1단계: "## 5. 결론" 섹션까지만 유지 (섹션 5 이후 모든 내용 제거)
    if "## 5. 결론" in text:
        parts = text.split("## 5. 결론")
        before_conclusion = parts[0]
        conclusion_raw = parts[1] if len(parts) > 1 else ""
        
        # 결론 내용에서 첫 번째 문단만 추출 (# 으로 시작하는 새 섹션 전까지)
        conclusion_lines = []
        seen_content = set()
        
        for line in conclusion_raw.split('\n'):
            stripped = line.strip()
            
            # 새 보고서 시작 패턴 (# 20XX년 ...)
            if re.match(r'^#\s*20\d{2}년', stripped):
                break
            
            # 새 섹션/새 보고서 시작이면 중단
            if stripped.startswith("# ") or stripped.startswith("## 6"):
                break
            
            # 잘못된 패턴 제거 (# 전력수요, # 결론 등)
            if stripped.startswith("# ") and "전력수요" in stripped:
                break
            if stripped.startswith("# ") and "의견" in stripped:
                break
            
            # 마크다운 코드블록 시작이면 중단
            if stripped.startswith("```"):
                break
            
            # 중복 문장 스킵 (첫 50자로 비교)
            content_key = stripped[:50] if len(stripped) > 50 else stripped
            if content_key and content_key in seen_content:
                continue
            
            if content_key:
                seen_content.add(content_key)
            
            conclusion_lines.append(line)
        
        # 결론 정리 (마지막 불완전 문장 제거)
        conclusion_text = '\n'.join(conclusion_lines).strip()
        
        # 인라인으로 붙은 새 보고서 제목 제거 (예: "...입니다. # 2025년 6월 전력수요")
        conclusion_text = re.sub(r'\s*#\s*20\d{2}년.*$', '', conclusion_text, flags=re.DOTALL)
        
        # 마지막 문장이 불완전하면 제거 (마침표/느낌표로 끝나지 않음)
        if conclusion_text and not conclusion_text.rstrip().endswith(('.', '!', '다.')):
            # 마지막 완전한 문장까지만 유지
            last_period = conclusion_text.rfind('.')
            if last_period > 0:
                conclusion_text = conclusion_text[:last_period + 1]
        
        text = before_conclusion + "## 5. 결론\n" + conclusion_text
    
    # 2단계: 불필요한 마커 제거
    text = text.replace('#end', '').replace('#END', '')
    text = re.sub(r'#\s*전력수요 예측 전문가 의견', '', text)
    
    # 3단계: 끝에 잘린 코드블록 제거
    if "```" in text:
        code_blocks = text.split("```")
        if len(code_blocks) % 2 == 0:
            text = "```".join(code_blocks[:-1])
    
    # 4단계: 끝에 불완전한 문장 제거
    text = text.rstrip()
    if text and not text.endswith(('.', '!', '?', '다.', '습니다.', '입니다.')):
        # 마지막 마침표 찾기
        last_period = text.rfind('.')
        if last_period > len(text) - 100:  # 마지막 100자 이내에 마침표가 있으면
            text = text[:last_period + 1]
    
    return text.strip()


def generate_with_llm(prompt: str, llm_url: str) -> Optional[str]:
    """LLM 서버에 보고서 생성 요청"""
    try:
        # vLLM 또는 OpenAI 호환 API 호출
        response = httpx.post(
            f"{llm_url}/v1/completions",
            json={
                "model": "./power_demand_merged_model",
                "prompt": prompt,
                "max_tokens": 1500,
                "temperature": 0.7,
                "repetition_penalty": 1.15,
                "stop": ["## 6.", "\n\n\n"],
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        text = result.get("choices", [{}])[0].get("text", "")
        # 후처리: 반복 제거
        return remove_repetitions(text)
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] LLM 서버 호출 실패: {e}")
        print(f"[ERROR] 응답 내용: {e.response.text}")
        return None
    except httpx.HTTPError as e:
        print(f"[ERROR] HTTP 에러: {e}")
        return None
    except Exception as e:
        import traceback
        print(f"[ERROR] 예외 발생: {e}")
        traceback.print_exc()
        return None


def save_report(
    content: str,
    year: int,
    month: int,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """보고서를 파일로 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{year}_{month:02d}{suffix}_{timestamp}.md"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="전력수요 보고서 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 프롬프트만 생성
  python generate_report.py --year 2025 --month 12

  # LLM으로 보고서 생성
  python generate_report.py --year 2025 --month 12 --llm-url http://localhost:8000

  # JSON 데이터만 출력
  python generate_report.py --year 2025 --month 12 --json
        """,
    )
    parser.add_argument("--year", type=int, required=True, help="대상 연도")
    parser.add_argument("--month", type=int, required=True, help="대상 월 (1-12)")
    parser.add_argument("--llm-url", type=str, default=None, help="LLM 서버 URL (없으면 프롬프트만 생성)")
    parser.add_argument("--output", type=Path, default=Path("./reports"), help="출력 디렉토리")
    parser.add_argument("--json", action="store_true", help="JSON 데이터만 출력")
    parser.add_argument("--prompt-only", action="store_true", help="프롬프트만 출력 (파일 저장 안함)")

    args = parser.parse_args()

    # 월 유효성 검사
    if not 1 <= args.month <= 12:
        print("[ERROR] 월은 1-12 사이여야 합니다.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"전력수요 보고서 생성: {args.year}년 {args.month}월")
    print(f"{'='*60}\n")

    # 1. 데이터 조회
    print("[1/5] 데이터 조회 중...")
    tools = CombinedTools()
    data = tools.get_report_data(args.year, args.month)

    if data.get("summary", {}).get("error"):
        print(f"[ERROR] {data['summary']['error']}")
        sys.exit(1)

    print(f"  - 전력수요: 최대 {data['summary']['max_demand']/10000:.1f}만kW")
    if data.get("weather") and not data["weather"].get("error"):
        print(f"  - 기상: 평균 {data['weather']['temperature_avg']:.1f}°C")
    else:
        print(f"  - 기상: 데이터 없음")

    # JSON 모드
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return

    # 2. 차트 생성 (다년간 비교 포함)
    print("[2/5] 차트 생성 중...")
    
    # 과거 연도별 동월 데이터 조회 (최대 10년, 요청 연도까지만)
    historical = tools.get_historical_demand(args.month, years=10)
    if historical:
        # 요청 연도까지의 데이터만 필터링 (미래 데이터 제외)
        historical = [h for h in historical if h['year'] <= args.year]
        if historical:
            years_range = f"{historical[-1]['year']}-{historical[0]['year']}"
            print(f"  - 히스토리컬 데이터: {args.month}월 ({years_range}, {len(historical)}년간)")
    
    chart_path = generate_weekly_chart(
        data.get("weekly_demand", []),
        args.year,
        args.month,
        args.output,
        historical_data=historical if historical else None,
    )
    if chart_path:
        print(f"  - 차트 저장: {chart_path}")
    else:
        print(f"  - 차트 생성 실패 (데이터 부족)")

    # 3. 프롬프트 생성
    print("[3/5] 프롬프트 생성 중...")
    prompt = build_prompt(data)

    if args.prompt_only:
        print("\n" + "="*60)
        print(prompt)
        print("="*60)
        return

    # 4. LLM 호출 또는 프롬프트 저장
    if args.llm_url:
        print(f"[4/5] LLM 서버 호출 중... ({args.llm_url})")
        report = generate_with_llm(prompt, args.llm_url)

        if report:
            # 차트를 보고서에 삽입
            if chart_path:
                chart_section = f"\n\n## 6. 주별 전력수요 추이 그래프\n\n![주별 전력수요 추이](./charts/{chart_path.name})\n"
                report = report + chart_section
            
            print("[5/5] 보고서 저장 중...")
            filepath = save_report(report, args.year, args.month, args.output, "_llm")
            print(f"\n[완료] 보고서 저장: {filepath}")
        else:
            print("[WARN] LLM 생성 실패. 프롬프트만 저장합니다.")
            filepath = save_report(prompt, args.year, args.month, args.output, "_prompt")
            print(f"\n[완료] 프롬프트 저장: {filepath}")
    else:
        print("[4/5] LLM 서버 미지정. 프롬프트만 저장합니다.")
        print("[5/5] 프롬프트 저장 중...")
        filepath = save_report(prompt, args.year, args.month, args.output, "_prompt")
        print(f"\n[완료] 프롬프트 저장: {filepath}")

    # 요약 출력
    print(f"\n{'='*60}")
    print("데이터 요약:")
    print(f"  최대부하: {data['summary']['max_demand']/10000:.1f}만kW")
    print(f"  평균부하: {data['summary']['avg_demand']/10000:.1f}만kW")
    print(f"  전년비: {data['summary'].get('yoy_change', 'N/A')}")
    if chart_path:
        print(f"  차트: {chart_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
