"""
시스템 테스트 스크립트
mcp_server/tools.py의 SQLite 직접 조회 사용
"""

from pathlib import Path
import sys

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mcp_tools():
    """MCP 도구 테스트"""
    print("\n[1] MCP 도구 (SQLite 직접 조회) 테스트")
    print("-" * 40)

    from report_agent.mcp_server.tools import CombinedTools

    tools = CombinedTools()

    # 전력수요 요약 조회
    summary = tools.get_demand_summary(2024, 8)
    if "error" in summary:
        print(f"  오류: {summary['error']}")
        return False

    print(f"  2024년 8월 전력수요:")
    print(f"    최대부하: {summary.get('max_demand', 0):,.0f} kW")
    print(f"    평균부하: {summary.get('avg_demand', 0):,.0f} kW")
    print(f"    전년비: {summary.get('yoy_change', 0):+.1f}%")

    # 주별 조회
    weekly = tools.get_weekly_demand(2024, 8)
    print(f"  주별 데이터: {len(weekly)}건")

    # 과거 데이터 조회
    historical = tools.get_historical_demand(8, years=5)
    print(f"  과거 데이터: {len(historical)}건")

    # 최대부하 상세
    peak = tools.get_peak_load(2024, 8)
    if not peak.get("error"):
        print(f"  최대부하 발생: {peak.get('peak_date')} {peak.get('peak_hour')}시")

    # 기상 조회
    weather = tools.get_weather_summary(2024, 8)
    if not weather.get("error"):
        print(f"  평균기온: {weather.get('temperature_avg', 0):.1f}°C")
    else:
        print(f"  기상: {weather.get('error')}")

    print("  OK: MCP 도구 정상")
    return True


def test_report_generator():
    """보고서 생성기 테스트"""
    print("\n[2] 보고서 생성기 테스트")
    print("-" * 40)

    from report_agent.report_generator import ReportGenerator

    generator = ReportGenerator()

    result = generator.generate_report(2024, 8)

    if not result["success"]:
        print(f"  오류: {result.get('error')}")
        return False

    print(f"  데이터 조회 완료: {result['year']}년 {result['month']}월")
    print(f"  프롬프트 길이: {len(result['prompt'])} 문자")
    print(f"  프롬프트 미리보기:")
    print("  " + "-" * 50)
    preview = result["prompt"][:500].replace("\n", "\n  ")
    print(f"  {preview}...")
    print("  " + "-" * 50)

    print("  OK: 보고서 생성기 정상")
    return True


def test_combined_data():
    """통합 데이터 조회 테스트"""
    print("\n[3] 통합 데이터 조회 테스트")
    print("-" * 40)

    from report_agent.mcp_server.tools import CombinedTools

    tools = CombinedTools()

    # 통합 조회
    result = tools.get_report_data(2024, 8)

    assert "summary" in result, "summary 없음"
    assert "weekly_demand" in result, "weekly_demand 없음"
    assert "historical" in result, "historical 없음"
    assert "weather" in result, "weather 없음"

    print(f"  통합 조회 결과:")
    print(f"    전력수요: {result['summary'].get('max_demand', 0):,.0f} kW")
    print(f"    주별데이터: {len(result['weekly_demand'])}건")
    print(f"    과거데이터: {len(result['historical'])}건")

    print("  OK: 통합 데이터 조회 정상")
    return True


def run_all_tests():
    """전체 테스트 실행"""
    print("=" * 60)
    print("전력수요 보고서 시스템 테스트")
    print("=" * 60)

    results = []

    try:
        # 1. MCP 도구 테스트
        results.append(("MCP 도구", test_mcp_tools()))

        # 2. 보고서 생성기 테스트
        results.append(("보고서 생성기", test_report_generator()))

        # 3. 통합 데이터 조회 테스트
        results.append(("통합 데이터", test_combined_data()))

        print("\n" + "=" * 60)
        print("테스트 결과:")
        all_passed = True
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        print("=" * 60)
        if all_passed:
            print("모든 테스트 통과!")
        else:
            print("일부 테스트 실패!")
        return all_passed

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
