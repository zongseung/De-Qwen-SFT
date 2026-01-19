"""
전력수요 보고서 생성 시스템 메인

사용법:
    # 1. 샘플 데이터 생성
    python -m report_agent.main --init-data

    # 2. MCP 서버 실행
    python -m report_agent.main --server

    # 3. 보고서 생성
    python -m report_agent.main --generate --year 2024 --month 8
"""

import argparse
import asyncio
from pathlib import Path
from typing import Optional


def init_sample_data():
    """샘플 데이터 초기화"""
    from .data.sample_data import create_sample_data

    output_dir = Path(__file__).parent / "sample_data"
    demand_path, weather_path = create_sample_data(
        start_year=2020,
        end_year=2025,
        output_dir=output_dir
    )
    print(f"샘플 데이터 생성 완료!")
    print(f"  전력수요: {demand_path}")
    print(f"  기상: {weather_path}")


def run_mcp_server(host: str = "0.0.0.0", port: int = 8001):
    """MCP 서버 실행"""
    from .mcp_server.server import run_server

    sample_dir = Path(__file__).parent / "sample_data"
    demand_csv = sample_dir / "power_demand.csv"
    weather_csv = sample_dir / "weather.csv"

    if not demand_csv.exists():
        print("샘플 데이터가 없습니다. --init-data 먼저 실행하세요.")
        return

    print(f"MCP 서버 시작: http://{host}:{port}")
    run_server(host, port, demand_csv, weather_csv)


def generate_report(year: int, month: int, output_dir: Optional[Path] = None):
    """보고서 생성"""
    from .report_service import ReportGenerator

    generator = ReportGenerator()
    result = generator.generate_report(year, month)

    if result["success"]:
        print(f"\n{'='*60}")
        print(f"{year}년 {month}월 전력수요 보고서 생성")
        print(f"{'='*60}\n")

        # 데이터 요약
        data = result["data"]
        demand = data.get("demand", {})
        weather = data.get("weather", {})

        print("[조회된 데이터]")
        print(f"  최대부하: {demand.get('max_load', 0)/10000:.1f}만kW")
        print(f"  평균부하: {demand.get('avg_load', 0)/10000:.1f}만kW")
        print(f"  전년비 최대부하: {demand.get('yoy_max_change', 0):+.1f}%")
        print(f"  평균기온: {weather.get('avg_temp', 0):.1f}°C")

        print(f"\n[생성된 프롬프트]")
        print("-" * 60)
        print(result["prompt"][:1500] + "..." if len(result["prompt"]) > 1500 else result["prompt"])
        print("-" * 60)

        # 파일 저장
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = output_dir / f"prompt_{year}_{month:02d}.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(result["prompt"])
            print(f"\n프롬프트 저장: {prompt_file}")

    else:
        print(f"오류: {result.get('error')}")


def main():
    parser = argparse.ArgumentParser(description="전력수요 보고서 생성 시스템")
    parser.add_argument("--init-data", action="store_true", help="샘플 데이터 생성")
    parser.add_argument("--server", action="store_true", help="MCP 서버 실행")
    parser.add_argument("--generate", action="store_true", help="보고서 생성")
    parser.add_argument("--year", type=int, default=2024, help="대상 연도")
    parser.add_argument("--month", type=int, default=8, help="대상 월")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")
    parser.add_argument("--output", type=Path, default=None, help="출력 디렉토리")

    args = parser.parse_args()

    if args.init_data:
        init_sample_data()
    elif args.server:
        run_mcp_server(args.host, args.port)
    elif args.generate:
        generate_report(args.year, args.month, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
