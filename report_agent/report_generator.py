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
        }

    def _format_weekly(self, weekly: List[Dict]) -> List[Dict]:
        """주별 데이터 형식 변환"""
        result = []
        for w in weekly:
            result.append({
                "week": w.get("week"),
                "max_load": w.get("max_demand", 0),
                "avg_load": w.get("avg_demand", 0),
                "start_date": "",  # DB에서 제공되지 않음
                "end_date": "",
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
            })
        return result

    def build_prompt(self, data: Dict[str, Any]) -> str:
        """보고서 생성 프롬프트 구성"""
        year = data["year"]
        month = data["month"]
        demand = data.get("demand", {})
        weather = data.get("weather", {})
        historical = data.get("historical", [])
        weekly = data.get("weekly_demand", [])
        peak_load = data.get("peak_load", {})

        # 전년비 변동률 문자열
        yoy_max = demand.get("yoy_max_change", 0) or 0
        yoy_str = f"{yoy_max:+.1f}%" if yoy_max else "N/A"

        # 피크부하 정보
        peak_info = ""
        if peak_load.get("peak_datetime"):
            peak_info = f"- 최대부하 발생 시점: {peak_load.get('peak_datetime')}"

        # 주별 데이터 문자열
        weekly_str = ""
        for w in weekly:
            if w.get("max_load"):
                weekly_str += f"- {w.get('week')}주차: 최대 {w.get('max_load', 0)/10000:.1f}만kW, 평균 {w.get('avg_load', 0)/10000:.1f}만kW\n"
        if not weekly_str:
            weekly_str = "데이터 없음"

        # 과거 데이터 문자열
        hist_str = ""
        for h in historical:
            if h.get("max_load"):
                hist_str += f"- {h.get('year')}년: 최대 {h.get('max_load', 0)/10000:.1f}만kW, 평균 {h.get('avg_load', 0)/10000:.1f}만kW\n"
        if not hist_str:
            hist_str = "데이터 없음"

        # 기상 정보 (과거 기상전망 요약)
        temp_info = ""
        if weather.get("avg_temp"):
            temp_info = f"""- 평균기온: {weather.get('avg_temp', 0):.1f}°C
- 최고기온: {weather.get('max_temp', 0):.1f}°C
- 최저기온: {weather.get('min_temp', 0):.1f}°C
- 평균습도: {weather.get('humidity_avg', 0):.1f}%"""
        else:
            temp_info = "기상 데이터 없음"

        # 기상 데이터 존재 여부 플래그
        has_weather = bool(weather.get("avg_temp"))

        # 기상 섹션 동적 생성
        weather_section = ""
        if has_weather:
            weather_section = f"""1. 기상전망
Ϛ {month}월 기온 전망
○ (입력된 평균/최고/최저기온 및 습도 정보를 바탕으로, 수요에 영향을 줄 수 있는 '변동성/수준'을 매우 간결히 1~2문장으로 서술. 단정 금지)
Ϛ {month}월 기타(습도 등) 참고
○ (습도 데이터가 있으면 1문장, 없으면 생략)

"""

        prompt = f"""당신은 중앙전력관제센터 수요예측팀의 실무자입니다.
아래 입력 데이터만을 근거로 {year}년 {month}월 '전력수요 예측/전망' 보고서를 작성하세요.
출력물은 사용자가 제공한 예시 PDF의 문체/형식(간결한 항목형, 표 중심, 페이지 1~2 구성)을 최대한 모사해야 합니다.

[중요 제약]
- 근거 없는 원인 단정(예: 기후변화/산업회복 등) 금지. 데이터로 확인 가능한 범위에서만 표현.
- 모든 수치는 만kW 단위로 표기(소수점 1자리).
- 분량은 A4 2페이지를 넘지 않도록 간결하게 작성.
- 기상 데이터가 제공되지 않았거나 "기상 데이터 없음"이면 1. 기상전망 섹션 전체를 작성하지 말 것.
- 표는 가능한 한 PDF처럼 "단위/증감률" 정보를 함께 제시.

========================
## 입력 데이터
========================

[기본 수요 요약]
- 대상 기간: {year}년 {month}월
- 최대부하: {demand.get('max_load', 0)/10000:.1f}만kW (전년동월 대비 {yoy_str})
- 평균부하: {demand.get('avg_load', 0)/10000:.1f}만kW
- 최소부하: {demand.get('min_load', 0)/10000:.1f}만kW
{peak_info}

[주별 전력수요(또는 주별 최대/평균)]
{weekly_str}

[과거 동월 실적(최근 5개년)]
{hist_str}

[기상 요약(있을 때만)]
{temp_info}

========================
## 출력 형식 (PDF 모사)
========================

# {year}년 {month}월 전력수요 예측전망
(상단에 기관/팀 표기는 생략 가능. 대신 보고서 톤은 실무 문서처럼 유지)

{weather_section}2. 과거 전력수요 추이(최근 5개년)
Ϛ 최근 5개년 동월 실적 및 금월 요약
- 아래 표는 '단위: 만kW, %' 형식으로 작성
- 표에는 최소한 최대부하/평균부하 행을 포함
- 전년대비 증감률(%)은 입력값이 있으면 표기, 없으면 N/A로 표기

[단위: 만kW, %]
(여기에 hist_str 내용을 '표 형태'로 재구성하여 출력)

3. 전력수요 전망(또는 수요 현황 요약)
Ϛ 최대부하(금월): {demand.get('max_load', 0)/10000:.1f}만kW
○ 금월 주별 최대전력(데이터가 있을 때만 표로 제시, 없으면 이 문단 전체 생략)
- 주차/기간/최대부하(만kW) 형태의 표로 간단히 작성

Ϛ 평균부하(금월): {demand.get('avg_load', 0)/10000:.1f}만kW
○ (선택) 평균부하 산출/추정에 사용한 방법이 입력에 없으면 "복수 모형/지표를 종합" 정도로만 1문장 처리
※ 과도한 설명/학술적 서술 금지

[작성 시 유의]
- 문단은 길게 쓰지 말고, 'Ϛ' + '○' 중심으로 간결히 구성
- 표가 핵심이며, 서술은 표를 해석하는 1~2문장 이내로 제한
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
                        "model": "./power_demand_merged_model",
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
