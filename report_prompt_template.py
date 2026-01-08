"""
전력수요 보고서 생성 프롬프트 템플릿
"""

# 월간 보고서 생성 프롬프트 (2019년 형식 기반)
MONTHLY_REPORT_PROMPT = """
당신은 한국전력거래소의 전력수요 예측 전문가입니다.
다음 데이터를 기반으로 전력수요 월간 보고서를 작성해주세요.
주어진 데이터만 사용하고, 없는 정보는 생성하지 마세요.

## 입력 데이터

### 기본 정보
- 대상 기간: {year}년 {month}월

### 기상 전망
- 기온 전망: {temp_forecast}
- 강수량 전망: {precip_forecast}

### 과거 전력수요 실적 (최근 5개년 동월)
{historical_data}

### 전력수요 전망
- 최대부하 예측: {max_load}만kW (전년 동월 대비 {yoy_max_change}%)
- 평균부하 예측: {avg_load}만kW (전년 동월 대비 {yoy_avg_change}%)

### 주별 최대전력 예측
{weekly_data}

### 예측 방법론
{methodology}

## 보고서 형식 (아래 형식 그대로 작성)

# {year}년 {month:02d}월 전력수요 예측 전망

## 1. 기상전망

### {month}월 기온 전망
(기온 전망 내용 작성)

### {month}월 강수량 전망
(강수량 전망 내용 작성)

## 2. 과거 전력수요 추이

### 최근 5개년 실적 및 전망
(과거 데이터 표 형식으로 정리)

- **최대부하 (만kW)**:
  (연도별 최대부하 및 증감률)

- **평균부하 (만kW)**:
  (연도별 평균부하 및 증감률)

## 3. 전력수요 전망 결과

- 최대부하 예측: {max_load}만kW

### {year}년 {month}월 주별 최대전력
(주별 데이터 작성)

- 평균부하 예측: {avg_load}만kW

**방법론/산식**
(예측 방법론 설명)

---
전문적이고 객관적인 어조로 작성하세요.
"""

# 간단 버전 프롬프트 (짧은 보고서)
SIMPLE_REPORT_PROMPT = """
다음 전력수요 데이터를 기반으로 간략한 월간 보고서를 작성해주세요.

## 데이터
- 기간: {year}년 {month}월
- 최대부하: {max_load}만kW (전년 대비 {yoy_change}%)
- 평균부하: {avg_load}만kW
- 기온 전망: {temp_forecast}

## 보고서 (마크다운 형식으로 작성)
"""

# 주간 보고서 프롬프트
WEEKLY_REPORT_PROMPT = """
다음 데이터를 기반으로 주간 전력수요 보고서를 작성해주세요.

## 주간 데이터
- 대상 기간: {year}년 {month}월 {week}주차 ({date_range})
- 최대부하: {max_load}만kW
- 전주 대비: {wow_change}%
- 평균 기온: {avg_temp}°C
- 기온 특이사항: {temp_notes}

## 보고서 형식
1. 주간 전력수요 현황
2. 기상 영향 분석
3. 다음 주 전망
"""


def generate_monthly_report_prompt(data: dict) -> str:
    """
    월간 보고서 프롬프트 생성
    
    Args:
        data: {
            "year": 2024,
            "month": 8,
            "max_load": 9200,
            "avg_load": 7800,
            "prev_year_max": 8800,
            "yoy_change": 4.5,
            "weekly_data": "1주: 8900, 2주: 9100, 3주: 9200, 4주: 9000",
            "temp_forecast": "평년보다 높음",
            "precip_forecast": "평년과 비슷",
            "temp_vs_normal": "+2.3도",
            "special_notes": "폭염 특보 예상"
        }
    
    Returns:
        완성된 프롬프트 문자열
    """
    return MONTHLY_REPORT_PROMPT.format(**data)


def generate_simple_report_prompt(data: dict) -> str:
    """간단 보고서 프롬프트 생성"""
    return SIMPLE_REPORT_PROMPT.format(**data)


def save_report(content: str, year: int, month: int, output_dir: str = "./generated_reports", prefix: str = "power_demand_report") -> str:
    """
    보고서를 마크다운 파일로 저장
    
    Args:
        content: 보고서 내용
        year: 년도
        month: 월
        output_dir: 저장 디렉토리
        prefix: 파일명 접두사
    
    Returns:
        저장된 파일 경로
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{prefix}_{year}_{month:02d}_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 보고서 저장 완료: {filename}")
    return filename


# 사용 예시
if __name__ == "__main__":
    # 예시 데이터
    sample_data = {
        "year": 2024,
        "month": 8,
        "max_load": 9247,
        "avg_load": 7823,
        "prev_year_max": 8856,
        "yoy_change": 4.4,
        "weekly_data": """
- 1주차 (7/29~8/4): 8,950만kW
- 2주차 (8/5~8/11): 9,150만kW  
- 3주차 (8/12~8/18): 9,247만kW (월 최대)
- 4주차 (8/19~8/25): 9,100만kW
- 5주차 (8/26~9/1): 8,800만kW
""",
        "temp_forecast": "평년보다 1.5~2.5도 높을 전망",
        "precip_forecast": "평년과 비슷하거나 적음",
        "temp_vs_normal": "+2.1도",
        "special_notes": "- 8월 중순 폭염 특보 예상\n- 냉방수요 급증 대비 필요"
    }
    
    prompt = generate_monthly_report_prompt(sample_data)
    print("=" * 80)
    print("생성된 프롬프트:")
    print("=" * 80)
    print(prompt)

