"""
Pure MCP Server - 전력수요 데이터 서버 (MCP Protocol)

FastMCP를 사용한 순수 MCP 프로토콜 서버
"""

import json
from mcp.server import FastMCP

try:
    from .tools import CombinedTools, ChartTools
except ImportError:
    from tools import CombinedTools, ChartTools

# MCP 서버 생성
mcp = FastMCP("전력수요 MCP Server")

# 도구 인스턴스 (서버 시작 시 한 번만 생성)
_tools: CombinedTools | None = None
_chart_tools: ChartTools | None = None


def get_tools() -> CombinedTools:
    global _tools
    if _tools is None:
        _tools = CombinedTools()
    return _tools


def get_chart_tools() -> ChartTools:
    global _chart_tools
    if _chart_tools is None:
        _chart_tools = ChartTools()
    return _chart_tools


# === MCP Tools ===


@mcp.tool()
def get_demand_summary(year: int, month: int) -> str:
    """월간 전력수요 요약 조회 (최대/평균부하, 전년비)

    Args:
        year: 연도 (예: 2024)
        month: 월 (1-12)

    Returns:
        JSON 형식의 월간 전력수요 요약
    """
    tools = get_tools()
    result = tools.get_demand_summary(year, month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_weekly_demand(year: int, month: int) -> str:
    """주별 전력수요 데이터 조회

    Args:
        year: 연도
        month: 월 (1-12)

    Returns:
        JSON 형식의 주별 전력수요 데이터
    """
    tools = get_tools()
    result = tools.get_weekly_demand(year, month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_peak_load(year: int, month: int) -> str:
    """최대부하 상세 정보 조회 (발생일, 시간)

    Args:
        year: 연도
        month: 월 (1-12)

    Returns:
        JSON 형식의 최대부하 정보
    """
    tools = get_tools()
    result = tools.get_peak_load(year, month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_historical_demand(month: int, years: int = 5, target_year: int | None = None) -> str:
    """과거 N년간 동월 전력수요 조회

    Args:
        month: 조회할 월 (1-12)
        years: 조회할 연도 수 (기본: 5)
        target_year: 기준 연도 (이 연도까지의 데이터만 조회)

    Returns:
        JSON 형식의 과거 동월 데이터
    """
    tools = get_tools()
    result = tools.get_historical_demand(month, years, target_year)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_report_data(year: int, month: int) -> str:
    """보고서 생성용 전체 데이터 조회 (요약, 주차, 과거, 기상 등)

    Args:
        year: 연도
        month: 월 (1-12)

    Returns:
        JSON 형식의 보고서용 통합 데이터
    """
    tools = get_tools()
    result = tools.get_report_data(year, month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def forecast_weekly_demand(
    year: int,
    month: int,
    model: str = "lstm",
    include_next_month: bool = False
) -> str:
    """주차별 전력수요 예측 (ARIMA/Holt-Winters/LSTM/Ensemble)

    Args:
        year: 예측 대상 연도
        month: 예측 대상 월 (1-12)
        model: 사용할 모델 (arima, holt_winters, lstm, ensemble)
        include_next_month: True면 다음 달까지 예측 (LSTM 8주 모델 사용)

    Returns:
        JSON 형식의 주차별 예측 결과
    """
    tools = get_tools()
    result = tools.forecast_weekly_demand(year, month, model, include_next_month)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_yearly_monthly_demand(target_year: int, target_month: int, years: int = 5) -> str:
    """연도별 월별 수요 조회 (차트/표용)

    Args:
        target_year: 기준 연도
        target_month: 기준 월 (1-12)
        years: 조회할 연도 수 (기본: 5)

    Returns:
        JSON 형식의 연도별 월별 수요 데이터
    """
    tools = get_tools()
    result = tools.get_yearly_monthly_demand(target_year, target_month, years)
    return json.dumps({"data": result}, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_weekly_chart(year: int, month: int) -> str:
    """주별 전력수요 추이 차트 생성 (base64 PNG)

    Args:
        year: 연도
        month: 월 (1-12)

    Returns:
        JSON 형식의 차트 데이터 (base64 인코딩된 이미지 포함)
    """
    chart_tools = get_chart_tools()
    result = chart_tools.generate_weekly_chart(year, month, return_base64=True)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def generate_yearly_monthly_chart(
    target_year: int,
    target_month: int,
    years: int = 5
) -> str:
    """연도별 월별 수요 차트 생성 (base64 PNG)

    Args:
        target_year: 기준 연도
        target_month: 기준 월 (1-12)
        years: 표시할 연도 수 (기본: 5)

    Returns:
        JSON 형식의 차트 데이터 (base64 인코딩된 이미지 포함)
    """
    chart_tools = get_chart_tools()
    result = chart_tools.generate_yearly_monthly_chart(
        target_year, target_month, years,
        return_base64=True
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def get_weather_summary(year: int, month: int) -> str:
    """월간 기상 요약 조회

    Args:
        year: 연도
        month: 월 (1-12)

    Returns:
        JSON 형식의 기상 요약 (평균/최고/최저 기온, 습도)
    """
    tools = get_tools()
    result = tools.get_weather_summary(year, month)
    return json.dumps(result, ensure_ascii=False, indent=2)


# === 서버 실행 ===

def run_stdio():
    """stdio 모드로 MCP 서버 실행"""
    mcp.run()


def run_sse(host: str = "0.0.0.0", port: int = 8001):
    """SSE 모드로 MCP 서버 실행"""
    mcp.run(transport="sse", sse_path="/sse", host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전력수요 MCP Server (Pure MCP)")
    parser.add_argument("--mode", choices=["stdio", "sse"], default="stdio",
                        help="전송 모드 (stdio: 표준입출력, sse: HTTP SSE)")
    parser.add_argument("--host", default="0.0.0.0", help="SSE 모드 호스트")
    parser.add_argument("--port", type=int, default=8001, help="SSE 모드 포트")
    args = parser.parse_args()

    if args.mode == "stdio":
        print("Starting MCP Server (stdio mode)...")
        run_stdio()
    else:
        print(f"Starting MCP Server (SSE mode) on {args.host}:{args.port}...")
        run_sse(args.host, args.port)
