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


def build_report_prompt(data: Dict[str, Any]) -> str:
    """보고서 생성용 프롬프트 구성

    Args:
        data: CombinedTools.get_report_data() 반환값
    """
    summary = data.get("summary", {})
    weather = data.get("weather", {})
    weekly = data.get("weekly_demand", [])
    historical = data.get("historical", [])

    year = summary.get("year", 0)
    month = summary.get("month", 0)
    target_period_label = f"{year}년 {month}월"

    # 기상 데이터
    avg_temp = weather.get("temperature_avg", 0) or 0 if weather and not weather.get("error") else 0
    max_temp = weather.get("temperature_max", 0) or 0 if weather and not weather.get("error") else 0
    min_temp = weather.get("temperature_min", 0) or 0 if weather and not weather.get("error") else 0
    humidity = weather.get("humidity_avg", 0) or 0 if weather and not weather.get("error") else 0

    weather_temp_text = f"평균 {avg_temp:.1f}°C (최고 {max_temp:.1f}°C, 최저 {min_temp:.1f}°C)" if avg_temp else "(값 미제공)"
    weather_humidity_text = f"평균 습도 {humidity:.1f}%" if humidity else "(값 미제공)"

    # 과거 5개년 데이터 처리
    historical_sorted = sorted(historical, key=lambda x: x.get("year", 0))
    cols = []
    max_loads_with_yoy = []
    avg_loads_with_yoy = []

    for h in historical_sorted[-5:]:
        cols.append(f"{h.get('year', '')}년{month}월")
        max_val = h.get("max_demand", 0) or 0
        avg_val = h.get("avg_demand", 0) or 0
        max_yoy = h.get("max_yoy")
        avg_yoy = h.get("avg_yoy")

        if max_val:
            max_str = f"{max_val/10:,.0f}"
            if max_yoy is not None:
                max_str += f" ({max_yoy:+.1f}%)"
            max_loads_with_yoy.append(max_str)
        else:
            max_loads_with_yoy.append("(값 미제공)")

        if avg_val:
            avg_str = f"{avg_val/10:,.0f}"
            if avg_yoy is not None:
                avg_str += f" ({avg_yoy:+.1f}%)"
            avg_loads_with_yoy.append(avg_str)
        else:
            avg_loads_with_yoy.append("(값 미제공)")

    while len(cols) < 5:
        cols.insert(0, "(값 미제공)")
        max_loads_with_yoy.insert(0, "(값 미제공)")
        avg_loads_with_yoy.insert(0, "(값 미제공)")

    # 주차별 테이블 생성
    weekly_table_rows = []
    for w in weekly:
        week_label = f"{w.get('week', '')}주"
        date_range = w.get("date_range", "")
        if not date_range:
            start = w.get("start_date", "")
            end = w.get("end_date", "")
            if start and end:
                try:
                    from datetime import datetime as dt
                    s_date = dt.strptime(start, "%Y-%m-%d")
                    e_date = dt.strptime(end, "%Y-%m-%d")
                    date_range = f"({s_date.month}/{s_date.day}~{e_date.month}/{e_date.day})"
                except ValueError:
                    date_range = ""
        week_max = w.get("max_demand", 0) or 0
        weekly_table_rows.append({
            "label": week_label,
            "range": date_range,
            "max": f"{week_max/10:,.0f}" if week_max else "(값 미제공)"
        })

    week_count = len(weekly_table_rows)
    week_headers = [f"{w['label']}{w['range']}" for w in weekly_table_rows]
    week_values = [w['max'] for w in weekly_table_rows]

    if week_count > 0:
        header_row = "| 구분 | " + " | ".join(week_headers) + " |"
        align_row = "|---|" + "---:|" * week_count
        data_row = "| 최대전력(만kW) | " + " | ".join(week_values) + " |"
        weekly_table = f"{header_row}\n{align_row}\n{data_row}"
    else:
        weekly_table = "(주별 데이터 미제공)"

    prompt = f"""[INST]
너는 전력수요 전망 보고서를 마크다운으로 작성하는 전문가다.
아래 데이터를 기반으로 {target_period_label} 전력수요 전망 보고서를 작성하라.

## 입력 데이터

### 기상 데이터
- 기온: {weather_temp_text}
- 습도: {weather_humidity_text}

### 과거 5개년 실적
| 구분 | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | {cols[4]} |
|---|---:|---:|---:|---:|---:|
| 최대부하(만kW) | {max_loads_with_yoy[0]} | {max_loads_with_yoy[1]} | {max_loads_with_yoy[2]} | {max_loads_with_yoy[3]} | {max_loads_with_yoy[4]} |
| 평균부하(만kW) | {avg_loads_with_yoy[0]} | {avg_loads_with_yoy[1]} | {avg_loads_with_yoy[2]} | {avg_loads_with_yoy[3]} | {avg_loads_with_yoy[4]} |

### 주차별 전력수요 전망
{weekly_table}

## 보고서 구성 (이 순서대로 작성)

# 1. 기상전망
- {month}월 기온 전망 (입력된 기상 데이터 기반)
- {month}월 강수량 전망

# 2. 과거 전력수요 추이
- 최근 5개년 {month}월 실적 분석 설명
- 최대부하/평균부하 표 (증감률 포함)

# 3. 전력수요 전망결과
- {target_period_label} 주차별 최대전력 표

## 출력 규칙
- 마크다운 보고서 본문만 출력
- 표는 Markdown Table 형식
- 없는 데이터는 "(값 미제공)" 표기
[/INST]

# {target_period_label} 전력수요 전망 보고서

"""
    return prompt


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


def insert_chart_after_section2(report: str, chart_path: Path) -> str:
    """# 2. 과거 전력수요 추이 섹션 바로 아래에 실적그래프 삽입"""
    import re

    chart_markdown = f"\n\n### 실적그래프\n\n![최근 5개년 전력수요 추이](./charts/{chart_path.name})\n"

    # "# 3." 또는 "## 3." 시작 전에 그래프 삽입
    pattern = r'(# 3\.|## 3\.)'
    match = re.search(pattern, report)

    if match:
        insert_pos = match.start()
        return report[:insert_pos] + chart_markdown + "\n" + report[insert_pos:]
    else:
        # 섹션 3을 못 찾으면 마지막에 추가
        return report + chart_markdown


def remove_repetitions(text: str) -> str:
    """연속으로 반복되는 문장/단락 제거 및 불필요한 텍스트 정리"""
    import re

    if not text or not text.strip():
        return text

    # 1단계: "# 3. 전력수요 전망결과" 섹션 이후 반복되는 보고서 제거
    # 패턴: 새 보고서 시작 (# 20XX년 ...) 또는 # 4. 이후 섹션
    lines = text.split('\n')
    result_lines = []
    found_section_3 = False

    for line in lines:
        stripped = line.strip()

        # 섹션 3 발견
        if '# 3.' in stripped or '## 3.' in stripped:
            found_section_3 = True

        # 섹션 3 이후에 새 보고서 시작 패턴 감지 → 중단
        if found_section_3:
            # 새 보고서 시작 (# 20XX년 ...)
            if re.match(r'^#\s*20\d{2}년', stripped):
                break
            # 섹션 4 이상 시작
            if re.match(r'^#\s*4\.', stripped) or re.match(r'^##\s*4\.', stripped):
                break
            # [INST] 태그 반복
            if '[INST]' in stripped:
                break

        result_lines.append(line)

    text = '\n'.join(result_lines)

    # 2단계: 불필요한 마커 제거
    text = text.replace('#end', '').replace('#END', '')
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[/INST\]', '', text)

    # 3단계: 끝에 불완전한 테이블 행 제거
    lines = text.rstrip().split('\n')
    while lines and lines[-1].strip().startswith('|') and lines[-1].count('|') < 3:
        lines.pop()
    text = '\n'.join(lines)

    # 4단계: 끝에 불완전한 문장 제거
    text = text.rstrip()
    if text and not text.endswith(('.', '!', '?', '|', '다.', '습니다.', '입니다.')):
        last_period = text.rfind('.')
        last_table = text.rfind('|')
        last_complete = max(last_period, last_table)
        if last_complete > len(text) - 100:
            text = text[:last_complete + 1]

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
                "stop": ["# 4.", "## 4.", "[INST]", "\n\n\n\n"],
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
    
    # 과거 연도별 동월 데이터 조회 (최대 10년, 요청 연도까지)
    historical = tools.get_historical_demand(args.month, years=10, target_year=args.year)
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
    prompt = build_report_prompt(data)

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
            # 차트를 "# 2. 과거 전력수요 추이" 섹션 아래에 삽입
            if chart_path:
                report = insert_chart_after_section2(report, chart_path)
            
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