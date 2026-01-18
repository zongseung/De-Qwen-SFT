"""
Pure MCP Client - MCP 프로토콜 기반 클라이언트

MCP 서버와 stdio 또는 SSE로 통신
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client


class MCPClientPure:
    """순수 MCP 프로토콜 클라이언트"""

    def __init__(self, mode: str = "stdio", server_url: str = "http://localhost:8001/sse"):
        """
        Args:
            mode: 'stdio' 또는 'sse'
            server_url: SSE 모드일 때 서버 URL
        """
        self.mode = mode
        self.server_url = server_url
        self._session: Optional[ClientSession] = None
        self._read_stream = None
        self._write_stream = None

    @asynccontextmanager
    async def connect(self):
        """MCP 서버에 연결"""
        if self.mode == "stdio":
            # stdio 모드: 서버를 subprocess로 실행
            server_script = Path(__file__).parent / "mcp_server" / "mcp_pure_server.py"
            # venv의 Python 사용
            venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"
            python_cmd = str(venv_python) if venv_python.exists() else sys.executable
            server_params = StdioServerParameters(
                command=python_cmd,
                args=[str(server_script), "--mode", "stdio"],
                env=None
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    yield self
                    self._session = None
        else:
            # SSE 모드: HTTP SSE로 연결
            async with sse_client(self.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    yield self
                    self._session = None

    async def list_tools(self) -> List[Dict]:
        """사용 가능한 도구 목록 조회"""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")
        result = await self._session.list_tools()
        return [{"name": t.name, "description": t.description} for t in result.tools]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """도구 호출"""
        if not self._session:
            raise RuntimeError("Not connected to MCP server")
        result = await self._session.call_tool(name, arguments)
        # 결과 파싱 (JSON 문자열인 경우)
        if result.content and len(result.content) > 0:
            text = result.content[0].text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return None

    # === 편의 메서드 (기존 mcp_client.py 호환) ===

    async def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        """보고서용 통합 데이터 조회"""
        return await self.call_tool("get_report_data", {"year": year, "month": month})

    async def forecast_weekly_demand(
        self,
        year: int,
        month: int,
        model: str = "lstm",
        include_next_month: bool = False
    ) -> Dict[str, Any]:
        """주차별 전력수요 예측"""
        return await self.call_tool("forecast_weekly_demand", {
            "year": year,
            "month": month,
            "model": model,
            "include_next_month": include_next_month
        })

    async def get_historical_demand(
        self,
        month: int,
        years: int = 5,
        target_year: Optional[int] = None
    ) -> List[Dict]:
        """과거 동월 수요 데이터 조회"""
        args = {"month": month, "years": years}
        if target_year:
            args["target_year"] = target_year
        return await self.call_tool("get_historical_demand", args)

    async def get_yearly_monthly_demand(
        self,
        target_year: int,
        target_month: int,
        years: int = 5
    ) -> Dict[str, Any]:
        """연도별 월별 수요 조회"""
        return await self.call_tool("get_yearly_monthly_demand", {
            "target_year": target_year,
            "target_month": target_month,
            "years": years
        })

    async def generate_yearly_monthly_chart(
        self,
        target_year: int,
        target_month: int,
        years: int = 5
    ) -> Dict[str, Any]:
        """연도별 월별 차트 생성 (base64)"""
        return await self.call_tool("generate_yearly_monthly_chart", {
            "target_year": target_year,
            "target_month": target_month,
            "years": years
        })

    async def generate_weekly_chart(self, year: int, month: int) -> Dict[str, Any]:
        """주별 차트 생성 (base64)"""
        return await self.call_tool("generate_weekly_chart", {
            "year": year,
            "month": month
        })

    async def get_demand_summary(self, year: int, month: int) -> Dict[str, Any]:
        """월간 수요 요약"""
        return await self.call_tool("get_demand_summary", {
            "year": year,
            "month": month
        })

    async def get_weather_summary(self, year: int, month: int) -> Dict[str, Any]:
        """월간 기상 요약"""
        return await self.call_tool("get_weather_summary", {
            "year": year,
            "month": month
        })


# === 동기 래퍼 (기존 코드 호환용) ===

class MCPClientPureSync:
    """동기 래퍼 클래스"""

    def __init__(self, mode: str = "stdio", server_url: str = "http://localhost:8001/sse"):
        self.mode = mode
        self.server_url = server_url

    def _run(self, coro):
        """비동기 코루틴을 동기적으로 실행"""
        return asyncio.run(self._execute(coro))

    async def _execute(self, coro):
        """MCP 클라이언트 컨텍스트 내에서 코루틴 실행"""
        client = MCPClientPure(self.mode, self.server_url)
        async with client.connect():
            return await coro(client)

    def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        return self._run(lambda c: c.get_report_data(year, month))

    def forecast_weekly_demand(
        self,
        year: int,
        month: int,
        model: str = "lstm",
        include_next_month: bool = False
    ) -> Dict[str, Any]:
        return self._run(lambda c: c.forecast_weekly_demand(year, month, model, include_next_month))

    def get_historical_demand(
        self,
        month: int,
        years: int = 5,
        target_year: Optional[int] = None
    ) -> List[Dict]:
        return self._run(lambda c: c.get_historical_demand(month, years, target_year))

    def get_yearly_monthly_demand(
        self,
        target_year: int,
        target_month: int,
        years: int = 5
    ) -> Dict[str, Any]:
        return self._run(lambda c: c.get_yearly_monthly_demand(target_year, target_month, years))

    def generate_yearly_monthly_chart(
        self,
        target_year: int,
        target_month: int,
        years: int = 5
    ) -> Dict[str, Any]:
        return self._run(lambda c: c.generate_yearly_monthly_chart(target_year, target_month, years))


# === 테스트 코드 ===

async def test_client():
    """MCP 클라이언트 테스트"""
    print("=== MCP Client Test (stdio mode) ===\n")

    client = MCPClientPure(mode="stdio")

    async with client.connect():
        # 도구 목록 조회
        print("1. List tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description'][:50]}...")

        print("\n2. Get demand summary (2024-08):")
        summary = await client.get_demand_summary(2024, 8)
        print(f"  Max demand: {summary.get('max_demand')} MW")
        print(f"  YoY change: {summary.get('yoy_change')}%")

        print("\n3. Forecast weekly demand (2025-09, LSTM):")
        forecast = await client.forecast_weekly_demand(2025, 9, model="lstm")
        print(f"  Model: {forecast.get('model')}")
        for fc in forecast.get('forecasts', [])[:4]:
            print(f"  Week {fc['week']}: {fc['max_demand']} MW")

        print("\n4. Generate chart (base64):")
        chart = await client.generate_yearly_monthly_chart(2024, 8)
        if chart.get('success'):
            b64_len = len(chart.get('image_base64', ''))
            print(f"  Success! base64 length: {b64_len} chars")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_client())
