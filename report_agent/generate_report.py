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


def build_report_prompt(data: Dict[str, Any], forecast: Dict[str, Any] = None) -> str:
    """보고서 생성용 프롬프트 구성

    Args:
        data: CombinedTools.get_report_data() 반환값
        forecast: ForecastTools.forecast_weekly_demand() 반환값 (예측 데이터)
    """
    summary = data.get("summary", {})
    weather = data.get("weather", {})
    weekly = data.get("weekly_demand", [])
    historical = data.get("historical", [])
    historical_weather = data.get("historical_weather", {})
    forecast_data = forecast.get("forecasts", []) if forecast else []

    year = summary.get("year", 0)
    month = summary.get("month", 0)
    target_period_label = f"{year}년 {month}월"

    # 현재 월 기상 데이터
    avg_temp = weather.get("temperature_avg", 0) or 0 if weather and not weather.get("error") else 0
    max_temp = weather.get("temperature_max", 0) or 0 if weather and not weather.get("error") else 0
    min_temp = weather.get("temperature_min", 0) or 0 if weather and not weather.get("error") else 0
    humidity = weather.get("humidity_avg", 0) or 0 if weather and not weather.get("error") else 0

    weather_temp_text = f"평균 {avg_temp:.1f}°C (최고 {max_temp:.1f}°C, 최저 {min_temp:.1f}°C)" if avg_temp else "(값 미제공)"
    weather_humidity_text = f"평균 습도 {humidity:.1f}%" if humidity else "(값 미제공)"

    # 과거 동월 기상 데이터 (기상전망용)
    hist_weather_avg_temp = historical_weather.get("avg_temperature") if historical_weather and not historical_weather.get("error") else None
    hist_weather_avg_humidity = historical_weather.get("avg_humidity") if historical_weather and not historical_weather.get("error") else None
    hist_weather_years_count = historical_weather.get("years_count", 0) if historical_weather else 0

    # 과거 5개년 데이터 처리
    historical_sorted = sorted(historical, key=lambda x: x.get("year", 0))
    cols = []
    max_loads_with_yoy = []
    avg_loads_with_yoy = []

    # 예측 데이터에서 최대/평균 계산 (해당 년월용)
    forecast_max = None
    forecast_avg = None
    if forecast_data:
        forecast_values = [fc.get("max_demand", 0) for fc in forecast_data if fc.get("max_demand")]
        if forecast_values:
            forecast_max = max(forecast_values)  # 주차별 최대값 중 최대
            forecast_avg = sum(forecast_values) / len(forecast_values)  # 주차별 평균

    # 전년도 데이터 찾기 (증감률 계산용)
    prev_year_data = None
    for h in historical_sorted:
        if h.get('year') == year - 1:
            prev_year_data = h
            break

    for h in historical_sorted[-5:]:
        h_year = h.get('year', 0)
        cols.append(f"{h_year}년{month}월")

        # 해당 년월이면 예측값 사용
        if h_year == year and forecast_max:
            max_val = forecast_max
            avg_val = forecast_avg if forecast_avg else (h.get("avg_demand", 0) or 0)
            # 전년 대비 증감률 재계산
            if prev_year_data:
                prev_max = prev_year_data.get("max_demand", 0) or 0
                prev_avg = prev_year_data.get("avg_demand", 0) or 0
                max_yoy = ((max_val - prev_max) / prev_max * 100) if prev_max else None
                avg_yoy = ((avg_val - prev_avg) / prev_avg * 100) if prev_avg else None
            else:
                max_yoy = h.get("max_yoy")
                avg_yoy = h.get("avg_yoy")
        else:
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

    # 주차별 테이블 생성 (예측 데이터 사용)
    weekly_table_rows = []

    # 예측 데이터가 있으면 예측값 사용, 없으면 실적 데이터 사용
    if forecast_data:
        for fc in forecast_data:
            week_label = f"{fc.get('week', '')}주"
            date_range = fc.get("date_range", "")
            week_max = fc.get("max_demand", 0) or 0
            weekly_table_rows.append({
                "label": week_label,
                "range": date_range,
                "max": f"{week_max/10:,.0f}" if week_max else "(값 미제공)"
            })
    else:
        # fallback: 실적 데이터 사용
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
        data_row = "| 최대부하(만kW) | " + " | ".join(week_values) + " |"
        weekly_table = f"{header_row}\n{align_row}\n{data_row}"
    else:
        weekly_table = "(주별 데이터 미제공)"

    # 기상 데이터 유무에 따라 섹션 구성
    # 현재 월 기상 데이터 또는 과거 동월 기상 데이터가 있으면 기상전망 섹션 생성
    has_current_weather = weather_temp_text != "(값 미제공)"
    has_historical_weather = hist_weather_avg_temp is not None

    if has_current_weather:
        # 현재 월 기상 데이터가 있는 경우
        weather_section = f"""### 기상 데이터
- 기온: {weather_temp_text}
- 습도: {weather_humidity_text}

"""
        weather_instruction = f"""# 1. 기상전망
- 위 "기상 데이터"를 반드시 참고하여 작성
- 기온({weather_temp_text})과 습도({weather_humidity_text})를 언급
- 기온이 전력수요에 미치는 영향 설명 (예: 고온 시 냉방수요 증가, 저온 시 난방수요 증가)

# 2. 과거 전력수요 추이"""
        forecast_section_num = "3"
    elif has_historical_weather:
        # 과거 동월 기상 데이터만 있는 경우 (기상전망 작성용)
        weather_section = f"""### 과거 {hist_weather_years_count}개년 {month}월 기상 데이터
- 평균 기온: {hist_weather_avg_temp:.1f}°C
- 평균 습도: {hist_weather_avg_humidity:.1f}%

"""
        weather_instruction = f"""# 1. 기상전망
- 위 "과거 {hist_weather_years_count}개년 {month}월 기상 데이터"를 반드시 참고하여 작성
- 과거 평균 기온({hist_weather_avg_temp:.1f}°C)과 습도({hist_weather_avg_humidity:.1f}%)를 언급하고, 올해도 유사한 기온이 예상됨을 서술
- 기온이 전력수요에 미치는 영향 설명 (예: 고온 시 냉방수요 증가)

# 2. 과거 전력수요 추이"""
        forecast_section_num = "3"
    else:
        weather_section = ""
        weather_instruction = "# 1. 과거 전력수요 추이"
        forecast_section_num = "2"

    prompt = f"""[INST]
너는 전력수요 전망 보고서를 마크다운으로 작성하는 전문가다.
아래 데이터를 기반으로 {target_period_label} 전력수요 전망 보고서를 작성하라.

## 입력 데이터

{weather_section}### 과거 5개년 실적
| 구분 | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | {cols[4]} |
|---|---:|---:|---:|---:|---:|
| 최대부하(만kW) | {max_loads_with_yoy[0]} | {max_loads_with_yoy[1]} | {max_loads_with_yoy[2]} | {max_loads_with_yoy[3]} | {max_loads_with_yoy[4]} |
| 평균부하(만kW) | {avg_loads_with_yoy[0]} | {avg_loads_with_yoy[1]} | {avg_loads_with_yoy[2]} | {avg_loads_with_yoy[3]} | {avg_loads_with_yoy[4]} |

### 주차별 전력수요 전망
{weekly_table}

## 보고서 구성 (이 순서대로 작성)

{weather_instruction}
- 최근 5개년 {month}월 실적 분석 설명
- "[단위: 만kW, 증감률(%)]" 표기 후 최대부하/평균부하 표 작성
- 각 수치에 증감률 괄호 포함 (예: "8,546만kW (+10.6%)")

# {forecast_section_num}. 전력수요 전망결과
- **중요: 위 "주차별 전력수요 전망" 표의 예측값을 절대 수정하지 말고 그대로 사용할 것**
- "[단위: 만kW]" 표기 후 {target_period_label} 주차별 최대부하 표 작성
- 주차 헤더에 날짜 범위 포함 (예: "1주(8/1~8/4)")

## 출력 규칙
- 마크다운 보고서 본문만 출력
- 표는 Markdown Table 형식
- 과거 실적 표: 수치 뒤에 증감률 괄호 표기 (예: 8,546만kW (+10.6%))
- 표 위에 반드시 "[단위: 만kW]" 명시
- 주차별 표 헤더에 날짜 범위 표기 (예: 1주(8/1~8/4), 2주(8/5~8/11))
- 입력 데이터의 증감률을 그대로 사용
- **주차별 전망 표의 예측값은 입력 데이터의 값을 그대로 복사하여 사용 (수치 변경 금지)**
- 기상 데이터가 없으면 기상전망 섹션 생략
[/INST]

# {target_period_label} 전력수요 전망 보고서

"""
    return prompt


def generate_yearly_monthly_chart(
    yearly_data: Dict[int, List[Dict[str, Any]]],
    target_year: int,
    target_month: int,
    output_dir: Path,
) -> Optional[Path]:
    """연도별 월별 평균 수요 라인 차트 생성

    Args:
        yearly_data: 연도별 월별 수요 데이터 {year: [{month, avg_demand, max_demand}, ...]}
        target_year: 기준 연도
        target_month: 기준 월
        output_dir: 출력 디렉토리

    Returns:
        생성된 차트 파일 경로
    """
    if not yearly_data:
        return None

    # 차트 저장 디렉토리 생성
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 색상 팔레트 (연도별로 다른 색상)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    fig, ax = plt.subplots(figsize=(12, 6))

    # 연도를 오름차순으로 정렬
    sorted_years = sorted(yearly_data.keys())

    for idx, year in enumerate(sorted_years):
        monthly_data = yearly_data[year]
        months = [d['month'] for d in monthly_data]
        avg_demands = [d['avg_demand'] / 10000 if d['avg_demand'] else 0 for d in monthly_data]

        color = colors[idx % len(colors)]

        # 기준 연도는 굵은 선으로 강조
        if year == target_year:
            ax.plot(months, avg_demands, 'o-', color=color, linewidth=3, markersize=10,
                   label=f'{year}', zorder=10)
            # 데이터 레이블 추가
            for m, d in zip(months, avg_demands):
                ax.annotate(f'{d:.0f}', (m, d), textcoords="offset points",
                           xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=color)
        else:
            ax.plot(months, avg_demands, 'o--', color=color, linewidth=1.5, markersize=6,
                   label=f'{year}', alpha=0.7)

    # X축 설정 (1-12월)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # 기준 월 강조 (세로선)
    ax.axvline(x=target_month, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Avg Power Demand (10MW)', fontsize=12)
    ax.set_title(f'Monthly Avg Power Demand by Year ({sorted_years[0]}-{sorted_years[-1]})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', title='Year')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    # 파일 저장
    filename = f"yearly_monthly_demand_{target_year}_{target_month:02d}.png"
    filepath = charts_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def insert_chart_after_section2(report: str, chart_path: Path) -> str:
    """# 2. 과거 전력수요 추이 섹션 바로 아래에 실적그래프 삽입"""
    import re

    # 원하는 형식: ### 실적그래프 다음에 ## 2. 과거 전력수요 추이 제목과 이미지
    chart_markdown = f"""

### 실적그래프

## 2. 과거 전력수요 추이
![최근 5개년 전력수요 추이](./charts/{chart_path.name})

"""

    # "# 3." 또는 "## 3." 시작 전에 그래프 삽입
    pattern = r'(# 3\.|## 3\.)'
    match = re.search(pattern, report)

    if match:
        insert_pos = match.start()
        return report[:insert_pos] + chart_markdown + report[insert_pos:]
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
    parser.add_argument("--forecast-model", type=str, default="lstm",
                       choices=["arima", "holt_winters", "lstm", "ensemble"],
                       help="예측 모델 선택 (기본: lstm)")
    parser.add_argument("--include-next-month", action="store_true",
                       help="다음 달까지 예측 (LSTM 8주 모델 사용)")

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

    # 1-2. 주차별 예측 수행
    next_month_str = " + 다음달" if args.include_next_month else ""
    print(f"[1-2/5] 주차별 예측 수행 중... (모델: {args.forecast_model}{next_month_str})")
    forecast_result = tools.forecast_weekly_demand(
        args.year, args.month,
        model=args.forecast_model,
        include_next_month=args.include_next_month
    )

    if forecast_result.get("forecasts"):
        print(f"  - {len(forecast_result['forecasts'])}개 주차 예측 완료")
        for fc in forecast_result["forecasts"]:
            print(f"    {fc['week']}주{fc['date_range']}: {fc['max_demand']/10:,.0f}만kW")
    else:
        print(f"  - 예측 실패 (데이터 부족 또는 모델 오류)")

    # JSON 모드
    if args.json:
        data["forecast"] = forecast_result
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return

    # 2. 차트 생성 (연도별 월별 평균 수요)
    print("[2/5] 차트 생성 중...")

    # 연도별 월별 평균 수요 조회 (5년치, 기준 연도는 기준 월까지만)
    yearly_monthly_data = tools.get_yearly_monthly_demand(args.year, args.month, years=5)
    if yearly_monthly_data:
        years_list = sorted(yearly_monthly_data.keys())
        print(f"  - 연도별 월별 데이터: {years_list[0]}-{years_list[-1]} ({len(yearly_monthly_data)}년간)")

    chart_path = generate_yearly_monthly_chart(
        yearly_monthly_data,
        args.year,
        args.month,
        args.output,
    )
    if chart_path:
        print(f"  - 차트 저장: {chart_path}")
    else:
        print(f"  - 차트 생성 실패 (데이터 부족)")

    # 3. 프롬프트 생성
    print("[3/5] 프롬프트 생성 중...")
    prompt = build_report_prompt(data, forecast_result)

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