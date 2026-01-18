"""MCP HTTP client for report generation (data, forecast, charts)."""

from pathlib import Path
from typing import Any, Dict, Optional

import httpx


class MCPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        return self._post("/tools/get_report_data", {"year": year, "month": month})

    def forecast_weekly_demand(
        self,
        year: int,
        month: int,
        model: str = "lstm",
        include_next_month: bool = False,
    ) -> Dict[str, Any]:
        return self._post(
            "/tools/forecast_weekly_demand",
            {
                "year": year,
                "month": month,
                "model": model,
                "include_next_month": include_next_month,
            },
        )

    def get_historical_demand(
        self,
        month: int,
        years: int = 5,
        target_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._post(
            "/tools/get_historical_demand",
            {"month": month, "years": years, "target_year": target_year},
        )

    def get_yearly_monthly_demand(
        self,
        target_year: int,
        target_month: int,
        years: int = 5,
    ) -> Dict[str, Any]:
        return self._post(
            "/tools/get_yearly_monthly_demand",
            {
                "target_year": target_year,
                "target_month": target_month,
                "years": years,
            },
        )

    def generate_yearly_monthly_chart(
        self,
        target_year: int,
        target_month: int,
        years: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        return self._post(
            "/tools/generate_yearly_monthly_chart",
            {
                "target_year": target_year,
                "target_month": target_month,
                "years": years,
                "output_dir": str(output_dir),
            },
        )
