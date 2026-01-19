"""
전력수요 보고서 생성기

데이터 조회 → 프롬프트 생성 → SFT 모델 호출 → 보고서 반환
(mcp_server/tools.py의 CombinedTools 사용)
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import httpx

from .mcp_server.tools import CombinedTools


class ReportGenerator:
    """전력수요 보고서 생성기"""

    def __init__(
        self,
        llm_server_url: str = "http://localhost:8000",
    ):
        self.tools = CombinedTools()
        self.llm_url = llm_server_url

    def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        """보고서 생성용 데이터 조회"""
        # CombinedTools.get_report_data() 사용
        data = self.tools.get_report_data(year, month)
        
        # 키 이름 매핑 (호환성 유지)
        return {
            "year": year,
            "month": month,
            "demand": {
                "max_load": data["summary"].get("max_demand", 0),
                "avg_load": data["summary"].get("avg_demand", 0),
                "min_load": data["summary"].get("min_demand", 0),
                "yoy_max_change": data["summary"].get("yoy_change", 0),
                "yoy_avg_change": data["summary"].get("yoy_change", 0),
            },
            "weather": {
                "avg_temp": data["weather"].get("temperature_avg", 0) if not data["weather"].get("error") else 0,
                "max_temp": data["weather"].get("temperature_max", 0) if not data["weather"].get("error") else 0,
                "min_temp": data["weather"].get("temperature_min", 0) if not data["weather"].get("error") else 0,
                "humidity_avg": data["weather"].get("humidity_avg", 0) if not data["weather"].get("error") else 0,
            },
            "weekly_demand": self._format_weekly(data.get("weekly_demand", [])),
            "historical": self._format_historical(data.get("historical", [])),
            "weather_forecast": {},  # 예보 데이터는 별도 API 필요
            "peak_load": data.get("peak_load", {}),
            "monthly_forecast": data.get("monthly_forecast", {}),
        }

    def _format_weekly(self, weekly: List[Dict]) -> List[Dict]:
        """주별 데이터 형식 변환"""
        result = []
        for w in weekly:
            result.append({
                "week": w.get("week"),
                "week_label": w.get("week_label", f"{w.get('week', '')}주"),
                "max_load": w.get("max_demand", 0),
                "avg_load": w.get("avg_demand", 0),
                "start_date": w.get("start_date", ""),
                "end_date": w.get("end_date", ""),
                "date_range": w.get("date_range", ""),
            })
        return result

    def _format_historical(self, historical: List[Dict]) -> List[Dict]:
        """과거 데이터 형식 변환"""
        result = []
        for h in historical:
            result.append({
                "year": h.get("year"),
                "month": None,  # 동월 데이터
                "max_load": h.get("max_demand", 0),
                "avg_load": h.get("avg_demand", 0),
                "max_yoy": h.get("max_yoy"),
                "avg_yoy": h.get("avg_yoy"),
            })
        return result

    def build_prompt(self, data: Dict[str, Any]) -> str:
        """보고서 생성 프롬프트 구성 (새 템플릿 형식)"""
        year = data["year"]
        month = data["month"]
        demand = data.get("demand", {})
        weather = data.get("weather", {})
        historical = data.get("historical", [])
        weekly = data.get("weekly_demand", [])
        monthly_forecast = data.get("monthly_forecast", {}) or {}

        # TARGET_PERIOD_LABEL
        target_period_label = f"{year}년 {month}월"

        # ===== 1. 기상전망 =====
        avg_temp = weather.get("avg_temp", 0) or 0
        max_temp = weather.get("max_temp", 0) or 0
        min_temp = weather.get("min_temp", 0) or 0
        humidity = weather.get("humidity_avg", 0) or 0

        weather_temp_text = f"평균 {avg_temp:.1f}°C (최고 {max_temp:.1f}°C, 최저 {min_temp:.1f}°C)" if avg_temp else "(값 미제공)"
        weather_humidity_text = f"평균 습도 {humidity:.1f}%" if humidity else "(값 미제공)"

        # ===== 2. 과거 전력수요 추이 =====
        # HISTORICAL 데이터 처리 (오래된 순으로 정렬)
        historical_sorted = sorted(historical, key=lambda x: x.get("year", 0))

        # 5개년 데이터 준비 (최대부하, 평균부하에 전년대비 증가율 괄호 포함)
        cols = []
        max_loads_with_yoy = []
        avg_loads_with_yoy = []

        forecast_max = monthly_forecast.get("max_demand")
        forecast_avg = monthly_forecast.get("avg_demand")

        prev_year_data = None
        for h in historical_sorted:
            if h.get("year") == year - 1:
                prev_year_data = h
                break

        for h in historical_sorted[-5:]:  # 최근 5개년
            h_year = h.get("year", 0)
            cols.append(f"{h_year}년{month}월")
            if h_year == year and (forecast_max is not None or forecast_avg is not None):
                max_val = forecast_max if forecast_max is not None else (h.get("max_load", 0) or 0)
                avg_val = forecast_avg if forecast_avg is not None else (h.get("avg_load", 0) or 0)
                if prev_year_data:
                    prev_max = prev_year_data.get("max_load", 0) or 0
                    prev_avg = prev_year_data.get("avg_load", 0) or 0
                    max_yoy = round((max_val - prev_max) / prev_max * 100, 1) if prev_max else None
                    avg_yoy = round((avg_val - prev_avg) / prev_avg * 100, 1) if prev_avg else None
                else:
                    max_yoy = h.get("max_yoy")
                    avg_yoy = h.get("avg_yoy")
            else:
                max_val = h.get("max_load", 0) or 0
                avg_val = h.get("avg_load", 0) or 0
                max_yoy = h.get("max_yoy")
                avg_yoy = h.get("avg_yoy")

            # 최대부하 (증가율) - MW → 만kW 변환: /10
            if max_val:
                max_str = f"{max_val/10:,.0f}"
                if max_yoy is not None:
                    max_str += f" ({max_yoy:+.1f}%)"
                max_loads_with_yoy.append(max_str)
            else:
                max_loads_with_yoy.append("(값 미제공)")

            # 평균부하 (증가율) - MW → 만kW 변환: /10
            if avg_val:
                avg_str = f"{avg_val/10:,.0f}"
                if avg_yoy is not None:
                    avg_str += f" ({avg_yoy:+.1f}%)"
                avg_loads_with_yoy.append(avg_str)
            else:
                avg_loads_with_yoy.append("(값 미제공)")

        # 5개 미만일 경우 패딩
        while len(cols) < 5:
            cols.insert(0, "(값 미제공)")
            max_loads_with_yoy.insert(0, "(값 미제공)")
            avg_loads_with_yoy.insert(0, "(값 미제공)")

        # HIST_PERIOD_TEXT
        years_list = [h.get("year") for h in historical_sorted if h.get("year")]
        if years_list:
            hist_period_text = f"{min(years_list)}~{max(years_list)}"
        else:
            hist_period_text = "(값 미제공)"

        # GRAPH_CAPTION_TEXT
        graph_caption = f"최근 5개년 {month}월 전력수요 추이를 나타낸 그래프로, 최대부하와 평균부하의 변동 추이를 확인할 수 있음."

        # ===== 3. 전력수요 전망 결과 =====
        # WEEKLY_MAX_BLOCKS (주차별 최대전력)
        weekly_table_rows = []
        for w in weekly:
            week_label = w.get("week_label", f"{w.get('week', '')}주")
            week_range = w.get("date_range", "")
            week_max = w.get("max_load", 0) or 0
            weekly_table_rows.append({
                "label": week_label,
                "range": week_range,
                "max": f"{week_max/10:,.0f}" if week_max else "(값 미제공)"  # MW → 만kW
            })

        # 주별 테이블 헤더/데이터 생성
        week_count = len(weekly_table_rows)
        week_headers = []
        week_values = []
        for w in weekly_table_rows:
            week_headers.append(f"{w['label']}{w['range']}")
            week_values.append(w['max'])

        # 주별 테이블 문자열 생성
        if week_count > 0:
            header_row = "| 구분 | " + " | ".join(week_headers) + " |"
            align_row = "|---|" + "---:|" * week_count
            data_row = "| 최대전력(만kW) | " + " | ".join(week_values) + " |"
            weekly_table = f"{header_row}\n{align_row}\n{data_row}"
        else:
            weekly_table = "(주별 데이터 미제공)"

        # 기상 데이터 유무에 따라 섹션 구성
        has_weather = weather_temp_text != "(값 미제공)"

        if has_weather:
            weather_section = f"""### 기상 데이터
- 기온: {weather_temp_text}
- 습도: {weather_humidity_text}

"""
            weather_instruction = f"""# 1. 기상전망
- {month}월 기온 전망 (입력된 기상 데이터 기반)

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
- "[단위: 만kW]" 표기 후 {target_period_label} 주차별 최대전력 표 작성
- 주차 헤더에 날짜 범위 포함 (예: "1주(8/1~8/4)")
- include_next_month/True/False/설정 여부 같은 메타 설명 문장은 절대 작성하지 말 것
- 중복 표, 예시 표, 추가 설명 표를 생성하지 말 것 (입력 데이터 기반 표만 작성)

## 출력 규칙
- 마크다운 보고서 본문만 출력
- 표는 Markdown Table 형식
- 과거 실적 표: 수치 뒤에 증감률 괄호 표기 (예: 8,546만kW (+10.6%))
- 표 위에 반드시 "[단위: 만kW]" 명시
- 주차별 표 헤더에 날짜 범위 표기 (예: 1주(8/1~8/4), 2주(8/5~8/11))
- 입력 데이터의 증감률을 그대로 사용
- 기상 데이터가 없으면 기상전망 섹션 생략
- include_next_month/True/False/설정 여부 등 메타 문구 금지
- 입력 데이터에 없는 예시/중복 표 금지
[/INST]

# {target_period_label} 전력수요 전망 보고서

"""
        return prompt

    async def generate_report_async(
        self,
        year: int,
        month: int,
        report_type: str = "full"
    ) -> Dict[str, Any]:
        """보고서 생성 (비동기)"""
        # 1. 데이터 조회
        data = self.get_report_data(year, month)

        # 2. 프롬프트 생성
        prompt = self.build_prompt(data)

        # 3. SFT 모델 호출
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.llm_url}/v1/completions",
                    json={
                        "model": "./power_demand_merged_model_llama3",
                        "prompt": prompt,
                        "max_tokens": 2000,
                        "temperature": 0.7,
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
                return {
                    "success": True,
                    "report": result.get("choices", [{}])[0].get("text", ""),
                    "data": data,
                }
            except httpx.HTTPError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "data": data,
                    "prompt": prompt,
                }

    def generate_report(
        self,
        year: int,
        month: int,
        report_type: str = "full"
    ) -> Dict[str, Any]:
        """보고서 생성 (동기) - 프롬프트만 반환"""
        # 1. 데이터 조회
        data = self.get_report_data(year, month)

        # 2. 프롬프트 생성
        prompt = self.build_prompt(data)

        return {
            "success": True,
            "data": data,
            "prompt": prompt,
            "year": year,
            "month": month,
        }


def build_report_prompt(data: Dict[str, Any]) -> str:
    """보고서 생성용 프롬프트 구성 (standalone 함수)

    Args:
        data: CombinedTools.get_report_data() 반환값
              - summary: {year, month, max_demand, avg_demand, ...}
              - weather: {temperature_avg, temperature_max, ...}
              - weekly_demand: [{week, max_demand, avg_demand, ...}, ...]
              - historical: [{year, max_demand, avg_demand, max_yoy, avg_yoy}, ...]
    """
    summary = data.get("summary", {})
    weather = data.get("weather", {})
    weekly = data.get("weekly_demand", [])
    historical = data.get("historical", [])
    monthly_forecast = data.get("monthly_forecast", {}) or {}

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

    # 기상 데이터 유무에 따라 섹션 구성
    has_weather = weather_temp_text != "(값 미제공)"

    if has_weather:
        weather_section = f"""### 기상 데이터
- 기온: {weather_temp_text}
- 습도: {weather_humidity_text}

"""
        weather_instruction = f"""# 1. 기상전망
- {month}월 기온 전망 (입력된 기상 데이터 기반)

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
- "[단위: 만kW]" 표기 후 {target_period_label} 주차별 최대전력 표 작성
- 주차 헤더에 날짜 범위 포함 (예: "1주(8/1~8/4)")
- include_next_month/True/False/설정 여부 같은 메타 설명 문장은 절대 작성하지 말 것
- 중복 표, 예시 표, 추가 설명 표를 생성하지 말 것 (입력 데이터 기반 표만 작성)

## 출력 규칙
- 마크다운 보고서 본문만 출력
- 표는 Markdown Table 형식
- 과거 실적 표: 수치 뒤에 증감률 괄호 표기 (예: 8,546만kW (+10.6%))
- 표 위에 반드시 "[단위: 만kW]" 명시
- 주차별 표 헤더에 날짜 범위 표기 (예: 1주(8/1~8/4), 2주(8/5~8/11))
- 입력 데이터의 증감률을 그대로 사용
- 기상 데이터가 없으면 기상전망 섹션 생략
- include_next_month/True/False/설정 여부 등 메타 문구 금지
- 입력 데이터에 없는 예시/중복 표 금지
[/INST]

# {target_period_label} 전력수요 전망 보고서

"""
    return prompt


def generate_simple_prompt(year: int, month: int, data: Dict[str, Any]) -> str:
    """간단 보고서 프롬프트"""
    demand = data.get("demand", {})
    weather = data.get("weather", {})

    return f"""다음 전력수요 데이터를 기반으로 간략한 월간 보고서를 작성해주세요.

## 데이터
- 기간: {year}년 {month}월
- 최대부하: {demand.get('max_load', 0)/10000:.1f}만kW (전년 대비 {demand.get('yoy_max_change', 0):+.1f}%)
- 평균부하: {demand.get('avg_load', 0)/10000:.1f}만kW
- 평균기온: {weather.get('avg_temp', 0):.1f}°C

## 보고서 (마크다운 형식으로 작성)
"""
