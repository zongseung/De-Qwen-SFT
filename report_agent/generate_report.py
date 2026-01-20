#!/usr/bin/env python
"""
전력수요 보고서 생성 CLI

Usage:
    # 프롬프트만 생성 (LLM 없이)
    python generate_report.py --year 2025 --month 12

    # LLM 서버로 보고서 생성
    python generate_report.py --year 2025 --month 12 --llm-url http://localhost:8000

    # 순수 MCP 모드 (stdio)
    python generate_report.py --year 2025 --month 12 --mcp-mode stdio --llm-url http://localhost:8000

    # 출력 디렉토리 지정
    python generate_report.py --year 2025 --month 12 --output ./reports
"""

import argparse
import asyncio
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx
from mcp_client import MCPClient


def build_report_prompt(
    data: Dict[str, Any],
    forecast: Dict[str, Any] = None,
    include_next_month: bool = False,
) -> str:
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
    monthly_forecast = data.get("monthly_forecast", {}) or {}
    weather_forecast = data.get("weather_forecast", {}) or {}
    forecast_data = forecast.get("forecasts", []) if forecast else []

    year = summary.get("year", 0)
    month = summary.get("month", 0)
    target_period_label = f"{year}년 {month}월"
    next_year = year + 1 if month == 12 else year
    next_month = 1 if month == 12 else month + 1
    next_period_label = f"{next_month}월"

    # 기상 예측 데이터 (LSTM 기반)
    temp_pred = weather_forecast.get("temp_mean_pred")
    humidity_pred = weather_forecast.get("humidity_mean_pred")
    weather_insight = weather_forecast.get("insight")

    # 현재 월 기상 데이터 (fallback)
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

    # 월별 LSTM 예측값 (3개 모델 평균)
    forecast_max = monthly_forecast.get("max_demand")
    forecast_avg = monthly_forecast.get("avg_demand")

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
        if h_year == year and (forecast_max is not None or forecast_avg is not None):
            max_val = forecast_max if forecast_max is not None else (h.get("max_demand", 0) or 0)
            avg_val = forecast_avg if forecast_avg is not None else (h.get("avg_demand", 0) or 0)
            # 전년 대비 증감률 재계산
            if prev_year_data:
                prev_max = prev_year_data.get("max_demand", 0) or 0
                prev_avg = prev_year_data.get("avg_demand", 0) or 0
                max_yoy = round((max_val - prev_max) / prev_max * 100, 1) if prev_max else None
                avg_yoy = round((avg_val - prev_avg) / prev_avg * 100, 1) if prev_avg else None
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
    # 우선순위: 1) LSTM 예측값 2) 현재 월 데이터 3) 과거 동월 데이터
    has_weather_prediction = temp_pred is not None
    has_current_weather = weather_temp_text != "(값 미제공)"
    has_historical_weather = hist_weather_avg_temp is not None

    if has_weather_prediction:
        # LSTM 예측값이 있는 경우 (가장 우선)
        pred_temp_text = f"예측 평균 기온: {temp_pred}°C"
        pred_humidity_text = f"예측 평균 습도: {humidity_pred}%" if humidity_pred else ""
        insight_text = weather_insight if weather_insight else ""

        weather_section = f"""### 기상 전망 (LSTM 예측 기반)
- {pred_temp_text}
{f"- {pred_humidity_text}" if pred_humidity_text else ""}
{f"- 분석: {insight_text}" if insight_text else ""}

"""
        weather_instruction = f"""# 1. 기상전망
- 위 "기상 전망 (LSTM 예측 기반)" 데이터를 반드시 참고하여 작성
- 예측 기온({temp_pred}°C){"과 습도(" + str(humidity_pred) + "%)" if humidity_pred else ""}를 언급
- 제공된 분석 내용을 바탕으로 전력수요에 미치는 영향 설명
- 기존 인사이트: {insight_text if insight_text else "기온에 따른 냉난방 수요 변화 예상"}

# 2. 과거 전력수요 추이"""
        forecast_section_num = "3"
    elif has_current_weather:
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
| 구분 | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | **{cols[4]}** |
|---|---:|---:|---:|---:|---:|
| 최대부하(만kW) | {max_loads_with_yoy[0]} | {max_loads_with_yoy[1]} | {max_loads_with_yoy[2]} | {max_loads_with_yoy[3]} | **{max_loads_with_yoy[4]}** |
| 평균부하(만kW) | {avg_loads_with_yoy[0]} | {avg_loads_with_yoy[1]} | {avg_loads_with_yoy[2]} | {avg_loads_with_yoy[3]} | **{avg_loads_with_yoy[4]}** |

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
- include-next-month가 True면 표를 2개로 분리: "{month}월 최대수요 전망"과 "{next_period_label} 최대수요 전망" 제목을 각각 표 위에 추가
- include-next-month/True/False/설정 여부 같은 메타 설명 문장은 절대 작성하지 말 것
- 중복 표, 예시 표, 추가 설명 표를 생성하지 말 것 (입력 데이터 기반 표만 1~2개 작성)

## 출력 규칙
- 마크다운 보고서 본문만 출력
- 표는 Markdown Table 형식
- 과거 실적 표: 수치 뒤에 증감률 괄호 표기 (예: 8,546만kW (+10.6%))
- 표 위에 반드시 "[단위: 만kW]" 명시
- 주차별 표 헤더에 날짜 범위 표기 (예: 1주(8/1~8/4), 2주(8/5~8/11))
- 입력 데이터의 증감률을 그대로 사용
- **주차별 전망 표의 예측값은 입력 데이터의 값을 그대로 복사하여 사용 (수치 변경 금지)**
- 기상 데이터가 없으면 기상전망 섹션 생략
- include-next-month가 True면 전망 결과 표를 2개로 분리하고, 각 표 위에 "{month}월 최대수요 전망", "{next_period_label} 최대수요 전망" 제목을 명시
- include-next-month/True/False/설정 여부 등 메타 문구 금지
- 입력 데이터에 없는 예시/중복 표 금지
[/INST]

# {target_period_label} 전력수요 전망 보고서

"""
    return prompt


def _infer_month_from_range(date_range: str) -> Optional[int]:
    if not date_range:
        return None
    match = re.search(r'(\d{1,2})/\d{1,2}\s*~\s*(\d{1,2})/\d{1,2}', date_range)
    if not match:
        return None
    return int(match.group(1))


def _format_historical_table(
    summary: Dict[str, Any],
    historical: List[Dict[str, Any]],
    forecast_data: List[Dict[str, Any]],
    include_next_month: bool,
    monthly_forecast: Optional[Dict[str, Any]] = None,
) -> str:
    year = summary.get("year", 0)
    month = summary.get("month", 0)

    historical_sorted = sorted(historical, key=lambda x: x.get("year", 0))
    cols = []
    max_loads_with_yoy = []
    avg_loads_with_yoy = []

    monthly_forecast = monthly_forecast or {}
    forecast_max = monthly_forecast.get("max_demand")
    forecast_avg = monthly_forecast.get("avg_demand")

    prev_year_data = None
    for h in historical_sorted:
        if h.get("year") == year - 1:
            prev_year_data = h
            break

    for h in historical_sorted[-5:]:
        h_year = h.get("year", 0)
        cols.append(f"{h_year}년{month}월")

        if h_year == year and (forecast_max is not None or forecast_avg is not None):
            max_val = forecast_max if forecast_max is not None else (h.get("max_demand", 0) or 0)
            avg_val = forecast_avg if forecast_avg is not None else (h.get("avg_demand", 0) or 0)
            if prev_year_data:
                prev_max = prev_year_data.get("max_demand", 0) or 0
                prev_avg = prev_year_data.get("avg_demand", 0) or 0
                max_yoy = round((max_val - prev_max) / prev_max * 100, 1) if prev_max else None
                avg_yoy = round((avg_val - prev_avg) / prev_avg * 100, 1) if prev_avg else None
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

    table = [
        f"| 구분 | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | **{cols[4]}** |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 최대부하 | {max_loads_with_yoy[0]} | {max_loads_with_yoy[1]} | {max_loads_with_yoy[2]} | {max_loads_with_yoy[3]} | **{max_loads_with_yoy[4]}** |",
        f"| 평균부하 | {avg_loads_with_yoy[0]} | {avg_loads_with_yoy[1]} | {avg_loads_with_yoy[2]} | {avg_loads_with_yoy[3]} | **{avg_loads_with_yoy[4]}** |",
        "",
        "* [단위: 만kW, 증감률(%)]",
    ]
    return "\n".join(table)


def _format_weekly_table(rows: List[Dict[str, Any]], header_label: str = "주차") -> str:
    if not rows:
        return "(주별 데이터 미제공)"

    week_headers = []
    week_values = []
    for row in rows:
        week_label = f"{row.get('week', '')}주"
        date_range = row.get("date_range", "")
        week_headers.append(f"{week_label}{date_range}")
        week_max = row.get("max_demand", 0) or 0
        week_values.append(f"{week_max/10:,.0f}" if week_max else "(값 미제공)")

    header_row = "| " + header_label + " | " + " | ".join(week_headers) + " |"
    align_row = "|---|" + "---:|" * len(week_headers)
    data_row = "| 최대부하(만kW) | " + " | ".join(week_values) + " |"
    return "\n".join([header_row, align_row, data_row, "", "* [단위: 만kW]"])


def _format_forecast_tables(
    summary: Dict[str, Any],
    weekly: List[Dict[str, Any]],
    forecast_data: List[Dict[str, Any]],
    include_next_month: bool,
) -> str:
    year = summary.get("year", 0)
    month = summary.get("month", 0)

    rows = []
    if forecast_data:
        for fc in forecast_data:
            rows.append({
                "week": fc.get("week"),
                "date_range": fc.get("date_range", ""),
                "max_demand": fc.get("max_demand", 0),
                "month": fc.get("month"),
                "year": fc.get("year"),
            })
    else:
        for w in weekly:
            rows.append({
                "week": w.get("week"),
                "date_range": w.get("date_range", ""),
                "max_demand": w.get("max_demand", 0),
                "month": month,
                "year": year,
            })

    next_year = year + 1 if month == 12 else year
    next_month = 1 if month == 12 else month + 1

    current_rows = []
    next_rows = []
    for row in rows:
        row_month = row.get("month") or _infer_month_from_range(row.get("date_range", "")) or month
        row_year = row.get("year")
        if row_year is None:
            row_year = year + 1 if row_month < month else year

        if include_next_month and row_month == next_month and row_year == next_year:
            next_rows.append(row)
        else:
            current_rows.append(row)

    if include_next_month and next_rows:
        current_block = _format_weekly_table(current_rows)
        next_block = _format_weekly_table(next_rows)
        return "\n".join([
            f"### {month}월 최대수요 전망",
            current_block,
            "",
            f"### {next_month}월 최대수요 전망",
            next_block,
        ])

    return _format_weekly_table(current_rows)


def _replace_table_after_heading(report: str, heading_patterns: List[re.Pattern], new_block: str) -> str:
    if not report:
        return report

    lines = report.splitlines()
    heading_idx = None
    for i, line in enumerate(lines):
        if any(p.search(line) for p in heading_patterns):
            heading_idx = i
            break

    if heading_idx is None:
        return report

    table_start = None
    for i in range(heading_idx + 1, len(lines)):
        if lines[i].lstrip().startswith("|"):
            table_start = i
            break

    if table_start is None:
        insert_at = heading_idx + 1
        insert_lines = ["", *new_block.splitlines(), ""]
        lines[insert_at:insert_at] = insert_lines
        return "\n".join(lines)

    table_end = table_start
    while table_end + 1 < len(lines) and lines[table_end + 1].lstrip().startswith("|"):
        table_end += 1

    unit_end = table_end
    scan = table_end + 1
    while scan < len(lines) and lines[scan].strip() == "":
        scan += 1
    if scan < len(lines) and "단위" in lines[scan]:
        unit_end = scan

    lines[table_start:unit_end + 1] = new_block.splitlines()
    return "\n".join(lines)


def apply_table_fixes(
    report: str,
    data: Dict[str, Any],
    forecast_result: Dict[str, Any],
    include_next_month: bool,
) -> str:
    summary = data.get("summary", {})
    historical = data.get("historical", [])
    weekly = data.get("weekly_demand", [])
    monthly_forecast = data.get("monthly_forecast", {})
    forecast_data = forecast_result.get("forecasts", []) if forecast_result else []

    if "include_next_month=True" in report:
        report = "\n".join(
            line for line in report.splitlines()
            if "include_next_month=True" not in line
        ).strip()

    historical_table = _format_historical_table(
        summary,
        historical,
        forecast_data,
        include_next_month,
        monthly_forecast,
    )
    report = _replace_table_after_heading(
        report,
        [
            re.compile(r"^##\s*2\.\s*과거 전력수요 추이"),
            re.compile(r"^#\s*2\.\s*과거 전력수요 추이"),
            re.compile(r"^##\s*과거 전력수요 추이"),
            re.compile(r"^#\s*과거 전력수요 추이"),
        ],
        historical_table,
    )

    forecast_block = _format_forecast_tables(summary, weekly, forecast_data, include_next_month)
    report = _replace_table_after_heading(
        report,
        [
            re.compile(r"^##\s*3\.\s*전력수요\s*전망\s*결과"),
            re.compile(r"^##\s*3\.\s*전력수요\s*전망결과"),
            re.compile(r"^#\s*3\.\s*전력수요\s*전망\s*결과"),
            re.compile(r"^#\s*3\.\s*전력수요\s*전망결과"),
            re.compile(r"^##\s*전력수요\s*전망\s*결과"),
            re.compile(r"^##\s*전력수요\s*전망결과"),
            re.compile(r"^#\s*전력수요\s*전망\s*결과"),
            re.compile(r"^#\s*전력수요\s*전망결과"),
        ],
        forecast_block,
    )

    report = _strip_fenced_blocks(report)
    report = _remove_orphan_tables(report)
    # 월별 예측 섹션 추가 (LSTM, ARIMA, Holt-Winters 개별 예측값)
    report = add_monthly_forecast_section(report, data)
    return report


def _strip_fenced_blocks(report: str) -> str:
    """Remove fenced code blocks that cause duplicate tables."""
    lines = report.splitlines()
    cleaned = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _remove_orphan_tables(report: str) -> str:
    """Drop tables that are not the main '구분' or '주차' tables."""
    lines = report.splitlines()
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            header = table_lines[0]
            if ("구분" in header) or ("주차" in header):
                cleaned.extend(table_lines)
            continue
        cleaned.append(line)
        i += 1
    return "\n".join(cleaned).strip()


def add_monthly_forecast_section(
    report: str,
    data: Dict[str, Any],
) -> str:
    """보고서 맨 끝에 월별 예측 섹션 추가 (LSTM, ARIMA, Holt-Winters 개별 예측값)

    사용자 요청 양식:
    ## 10월 전력수요 전망 : 예측값(모델 3개의 평균)
    - LSTM: xxx 만kW
    - Holt-Winters: xxx 만kW
    - ARIMA: xxx 만kW
    """
    monthly_forecast = data.get("monthly_forecast", {})
    if not monthly_forecast:
        return report

    summary = data.get("summary", {})
    month = summary.get("month", 0)
    model_predictions = monthly_forecast.get("model_predictions", {})

    # 평균부하 예측 (mean)
    mean_preds = model_predictions.get("mean", {})
    mean_lstm = mean_preds.get("lstm")
    mean_hw = mean_preds.get("holt_winters")
    mean_arima = mean_preds.get("arima")
    mean_avg = monthly_forecast.get("avg_demand")

    # 최대부하 예측 (peak)
    peak_preds = model_predictions.get("peak", {})
    peak_lstm = peak_preds.get("lstm")
    peak_hw = peak_preds.get("holt_winters")
    peak_arima = peak_preds.get("arima")
    peak_avg = monthly_forecast.get("max_demand")

    # 섹션 문자열 생성
    section_lines = [
        "",
        f"## {month}월 전력수요 전망 : 예측값 (모델 3개의 평균)",
        "",
    ]

    # 평균부하
    if mean_avg:
        section_lines.append(f"### 평균부하 전망: {mean_avg/10:,.0f} 만kW")
        if mean_lstm:
            section_lines.append(f"- LSTM: {mean_lstm/10:,.0f} 만kW")
        if mean_hw:
            section_lines.append(f"- Holt-Winters: {mean_hw/10:,.0f} 만kW")
        if mean_arima:
            section_lines.append(f"- ARIMA: {mean_arima/10:,.0f} 만kW")
        section_lines.append("")

    # 최대부하
    if peak_avg:
        section_lines.append(f"### 최대부하 전망: {peak_avg/10:,.0f} 만kW")
        if peak_lstm:
            section_lines.append(f"- LSTM: {peak_lstm/10:,.0f} 만kW")
        if peak_hw:
            section_lines.append(f"- Holt-Winters: {peak_hw/10:,.0f} 만kW")
        if peak_arima:
            section_lines.append(f"- ARIMA: {peak_arima/10:,.0f} 만kW")

    return report.rstrip() + "\n" + "\n".join(section_lines)


def insert_chart_after_section2(report: str, chart_path: Path) -> str:
    """# 2. 과거 전력수요 추이 섹션 내부에 실적그래프 삽입"""
    import re

    chart_markdown = f"""
### 실적그래프
![최근 5개년 전력수요 추이](./charts/{chart_path.name})
"""

    lines = report.splitlines()
    section_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^##\s*2\.\s*과거 전력수요 추이", line) or re.match(r"^#\s*2\.\s*과거 전력수요 추이", line):
            section_idx = i
            break

    if section_idx is None:
        return report + "\n\n" + chart_markdown.strip() + "\n"

    insert_at = section_idx + 1
    insert_lines = ["", *chart_markdown.strip().splitlines(), ""]
    lines[insert_at:insert_at] = insert_lines
    return "\n".join(lines)


def remove_repetitions(text: str) -> str:
    """연속으로 반복되는 문장/단락 제거 및 불필요한 텍스트 정리"""
    import re

    if not text or not text.strip():
        return text

    # 0단계: "include-next-month" 관련 메타 문구 제거
    # LLM이 프롬프트 지시사항을 출력에 포함시키는 문제 해결
    text = re.sub(r'include[-_]?next[-_]?month[^\n]*\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[단위: 만kW\]\s*\n(\s*\n)+', '[단위: 만kW]\n\n', text)  # 중복 빈 줄 정리

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
                "model": "./power_demand_merged_model_llama3",
                "prompt": prompt,
                "max_tokens": 800,
                "temperature": 0.2, # 최대한 정확한 답변 유도 
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


async def run_with_mcp_stdio(args):
    """순수 MCP (stdio) 모드로 보고서 생성"""
    from mcp_client_pure import MCPClientPure

    print(f"\n{'='*60}")
    print(f"전력수요 보고서 생성: {args.year}년 {args.month}월 (MCP stdio 모드)")
    print(f"{'='*60}\n")

    client = MCPClientPure(mode="stdio")

    async with client.connect():
        # 1. 데이터 조회
        print("[1/5] 데이터 조회 중...")
        data = await client.get_report_data(args.year, args.month)

        if args.include_next_month:
            hist_resp = await client.get_historical_demand(args.month, years=6, target_year=args.year)
            if isinstance(hist_resp, list):
                data["historical"] = hist_resp

        if data.get("summary", {}).get("error"):
            print(f"[ERROR] {data['summary']['error']}")
            return

        print(f"  - 전력수요: 최대 {data['summary']['max_demand']/10000:.1f}만kW")
        if data.get("weather") and not data["weather"].get("error"):
            print(f"  - 기상: 평균 {data['weather']['temperature_avg']:.1f}°C")
        else:
            print(f"  - 기상: 데이터 없음")

        # 1-2. 주차별 예측 수행
        next_month_str = " + 다음달" if args.include_next_month else ""
        print(f"[1-2/5] 주차별 예측 수행 중... (모델: {args.forecast_model}{next_month_str})")
        forecast_result = await client.forecast_weekly_demand(
            args.year,
            args.month,
            model=args.forecast_model,
            include_next_month=args.include_next_month,
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

        # 2. 차트 생성 (base64로 받아서 파일로 저장)
        print("[2/5] 차트 생성 중...")
        chart_resp = await client.generate_yearly_monthly_chart(args.year, args.month, years=5)
        chart_path = None

        if chart_resp.get("success") and chart_resp.get("image_base64"):
            # base64 디코딩해서 파일로 저장
            charts_dir = args.output / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            filename = f"yearly_monthly_demand_{args.year}_{args.month:02d}.png"
            chart_path = charts_dir / filename

            img_data = base64.b64decode(chart_resp["image_base64"])
            with open(chart_path, "wb") as f:
                f.write(img_data)
            print(f"  - 차트 저장: {chart_path}")
        else:
            print(f"  - 차트 생성 실패: {chart_resp.get('error', '데이터 부족')}")

        # 3. 프롬프트 생성
        print("[3/5] 프롬프트 생성 중...")
        prompt = build_report_prompt(data, forecast_result, args.include_next_month)

        if args.prompt_only:
            print("\n" + "="*60)
            print(prompt)
            print("="*60)
            return

        # 4. LLM 호출 (클라이언트에서 직접 - 하이브리드 방식)
        if args.llm_url:
            print(f"[4/5] LLM 서버 호출 중... ({args.llm_url})")
            report = generate_with_llm(prompt, args.llm_url)

            if report:
                if chart_path:
                    report = insert_chart_after_section2(report, chart_path)
                report = apply_table_fixes(report, data, forecast_result, args.include_next_month)

                print("[5/5] 보고서 저장 중...")
                filepath = save_report(report, args.year, args.month, args.output, "_llm_mcp")
                print(f"\n[완료] 보고서 저장: {filepath}")
            else:
                print("[WARN] LLM 생성 실패. 프롬프트만 저장합니다.")
                filepath = save_report(prompt, args.year, args.month, args.output, "_prompt_mcp")
                print(f"\n[완료] 프롬프트 저장: {filepath}")
        else:
            print("[4/5] LLM 서버 미지정. 프롬프트만 저장합니다.")
            print("[5/5] 프롬프트 저장 중...")
            filepath = save_report(prompt, args.year, args.month, args.output, "_prompt_mcp")
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


def main():
    parser = argparse.ArgumentParser(
        description="전력수요 보고서 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 프롬프트만 생성
  python generate_report.py --year 2025 --month 12

  # LLM으로 보고서 생성 (HTTP 모드)
  python generate_report.py --year 2025 --month 12 --llm-url http://localhost:8000

  # 순수 MCP 모드 (stdio)
  python generate_report.py --year 2025 --month 12 --mcp-mode stdio --llm-url http://localhost:8000

  # JSON 데이터만 출력
  python generate_report.py --year 2025 --month 12 --json
        """,
    )
    parser.add_argument("--year", type=int, required=True, help="대상 연도")
    parser.add_argument("--month", type=int, required=True, help="대상 월 (1-12)")
    parser.add_argument("--llm-url", type=str, default=None, help="LLM 서버 URL (없으면 프롬프트만 생성)")
    parser.add_argument("--mcp-url", type=str, default="http://localhost:8001", help="MCP 서버 URL (HTTP 모드)")
    parser.add_argument("--mcp-mode", type=str, default="http", choices=["http", "stdio"],
                       help="MCP 통신 모드 (http: FastAPI, stdio: 순수 MCP)")
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

    # MCP 모드에 따라 분기
    if args.mcp_mode == "stdio":
        asyncio.run(run_with_mcp_stdio(args))
        return

    # === HTTP 모드 (기존 로직) ===
    print(f"\n{'='*60}")
    print(f"전력수요 보고서 생성: {args.year}년 {args.month}월")
    print(f"{'='*60}\n")

    # 1. 데이터 조회
    print("[1/5] 데이터 조회 중...")
    mcp = MCPClient(args.mcp_url)
    data = mcp.get_report_data(args.year, args.month)
    if args.include_next_month:
        hist_resp = mcp.get_historical_demand(args.month, years=6, target_year=args.year)
        if isinstance(hist_resp, dict) and "error" in hist_resp:
            print(f"[WARN] 과거 데이터 조회 실패: {hist_resp['error']}")
        else:
            data["historical"] = hist_resp

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
    forecast_result = mcp.forecast_weekly_demand(
        args.year,
        args.month,
        model=args.forecast_model,
        include_next_month=args.include_next_month,
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
    ym_resp = mcp.get_yearly_monthly_demand(args.year, args.month, years=5)
    yearly_monthly_data = ym_resp.get("data", {})
    if yearly_monthly_data:
        years_list = sorted(yearly_monthly_data.keys())
        print(f"  - 연도별 월별 데이터: {years_list[0]}-{years_list[-1]} ({len(yearly_monthly_data)}년간)")

    chart_resp = mcp.generate_yearly_monthly_chart(args.year, args.month, years=5, output_dir=args.output)
    chart_path = None
    if chart_resp.get("success"):
        chart_path = Path(chart_resp["filepath"])
        print(f"  - 차트 저장: {chart_path}")
    else:
        print(f"  - 차트 생성 실패: {chart_resp.get('error', '데이터 부족')}")

    # 3. 프롬프트 생성
    print("[3/5] 프롬프트 생성 중...")
    prompt = build_report_prompt(data, forecast_result, args.include_next_month)

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
            report = apply_table_fixes(report, data, forecast_result, args.include_next_month)
            
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
