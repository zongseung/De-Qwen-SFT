"""
MCP Server - 전력수요 데이터 서버

SQLite DB에서 직접 조회하여 API로 제공
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

try:
    from .tools import CombinedTools, DemandTools, ChartTools
except ImportError:
    from tools import CombinedTools, DemandTools, ChartTools


# Request 스키마
class MonthlyRequest(BaseModel):
    year: int
    month: int


class HistoricalRequest(BaseModel):
    month: int
    years: int = 5
    target_year: int | None = None


class ForecastRequest(BaseModel):
    year: int
    month: int
    model: str = "lstm"
    include_next_month: bool = False


class YearlyMonthlyRequest(BaseModel):
    target_year: int
    target_month: int
    years: int = 5


class YearlyMonthlyChartRequest(BaseModel):
    target_year: int
    target_month: int
    years: int = 5
    output_dir: str = "./reports"


def create_mcp_server() -> FastAPI:
    """MCP 서버 생성"""

    app = FastAPI(
        title="전력수요 MCP Server",
        description="전력수요 데이터 조회 API (SQLite DB)",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 도구 초기화
    tools = CombinedTools()
    chart_tools = ChartTools()

    # === Health Check ===
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "demand-mcp-server"}

    # === MCP Tool Endpoints ===

    @app.post("/tools/get_demand_summary")
    async def get_demand_summary(req: MonthlyRequest) -> Dict[str, Any]:
        """월간 전력수요 요약 조회"""
        result = tools.get_demand_summary(req.year, req.month)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/tools/get_weekly_demand")
    async def get_weekly_demand(req: MonthlyRequest) -> List[Dict[str, Any]]:
        """주별 전력수요 조회"""
        return tools.get_weekly_demand(req.year, req.month)

    @app.post("/tools/get_peak_load")
    async def get_peak_load(req: MonthlyRequest) -> Dict[str, Any]:
        """최대부하 상세 정보 조회"""
        result = tools.get_peak_load(req.year, req.month)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.post("/tools/get_historical_demand")
    async def get_historical_demand(req: HistoricalRequest) -> List[Dict[str, Any]]:
        """과거 N년간 동월 데이터 조회"""
        return tools.get_historical_demand(req.month, req.years, target_year=req.target_year)

    @app.post("/tools/get_report_data")
    async def get_report_data(req: MonthlyRequest) -> Dict[str, Any]:
        """보고서 생성용 전체 데이터 조회"""
        return tools.get_report_data(req.year, req.month)

    @app.post("/tools/forecast_weekly_demand")
    async def forecast_weekly_demand(req: ForecastRequest) -> Dict[str, Any]:
        """주차별 전력수요 예측 (ARIMA/HW/LSTM, 다음달 포함 가능)"""
        return tools.forecast_weekly_demand(
            req.year,
            req.month,
            model=req.model,
            include_next_month=req.include_next_month,
        )

    @app.post("/tools/get_yearly_monthly_demand")
    async def get_yearly_monthly_demand(req: YearlyMonthlyRequest) -> Dict[str, Any]:
        """연도별 월별 수요 조회 (차트/표용)"""
        data = tools.get_yearly_monthly_demand(req.target_year, req.target_month, req.years)
        return {"data": data}

    @app.post("/tools/generate_yearly_monthly_chart")
    async def generate_yearly_monthly_chart(req: YearlyMonthlyChartRequest) -> Dict[str, Any]:
        """연도별 월별 수요 차트 생성 (PNG)"""
        result = chart_tools.generate_yearly_monthly_chart(
            target_year=req.target_year,
            target_month=req.target_month,
            years=req.years,
            output_dir=req.output_dir,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "차트 생성 실패"))
        return result

    @app.post("/tools/generate_weekly_chart")
    async def generate_weekly_chart(req: MonthlyRequest) -> Dict[str, Any]:
        """주별 전력수요 추이 차트 생성"""
        result = chart_tools.generate_weekly_chart(req.year, req.month)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "차트 생성 실패"))
        return result

    # === MCP Protocol Endpoints ===

    @app.get("/mcp/tools")
    async def list_tools():
        """사용 가능한 MCP 도구 목록"""
        return {
            "tools": [
                {
                    "name": "get_demand_summary",
                    "description": "월간 전력수요 요약 (최대/평균부하, 전년비)",
                    "parameters": {"year": "int", "month": "int"},
                },
                {
                    "name": "get_weekly_demand",
                    "description": "주별 전력수요 데이터",
                    "parameters": {"year": "int", "month": "int"},
                },
                {
                    "name": "get_peak_load",
                    "description": "최대부하 상세 정보 (발생일, 시간)",
                    "parameters": {"year": "int", "month": "int"},
                },
                {
                    "name": "get_historical_demand",
                    "description": "과거 N년간 동월 전력수요",
                    "parameters": {"month": "int", "years": "int (default: 5)", "target_year": "int | None"},
                },
                {
                    "name": "get_report_data",
                    "description": "보고서 생성용 전체 데이터",
                    "parameters": {"year": "int", "month": "int"},
                },
                {
                    "name": "forecast_weekly_demand",
                    "description": "주차별 전력수요 예측 (ARIMA/HW/LSTM, 다음달 포함 가능)",
                    "parameters": {
                        "year": "int",
                        "month": "int",
                        "model": "str (arima|holt_winters|lstm|ensemble, default: lstm)",
                        "include_next_month": "bool",
                    },
                },
                {
                    "name": "get_yearly_monthly_demand",
                    "description": "연도별 월별 수요 조회 (차트/표용)",
                    "parameters": {
                        "target_year": "int",
                        "target_month": "int",
                        "years": "int (default: 5)",
                    },
                },
                {
                    "name": "generate_yearly_monthly_chart",
                    "description": "연도별 월별 수요 차트 생성 (PNG 파일)",
                    "parameters": {
                        "target_year": "int",
                        "target_month": "int",
                        "years": "int (default: 5)",
                        "output_dir": "str (default: ./reports)",
                    },
                },
                {
                    "name": "generate_weekly_chart",
                    "description": "주별 전력수요 추이 라인 차트 생성 (PNG 파일)",
                    "parameters": {"year": "int", "month": "int"},
                },
            ]
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 8001):
    """서버 실행"""
    app = create_mcp_server()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전력수요 MCP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    run_server(args.host, args.port)
