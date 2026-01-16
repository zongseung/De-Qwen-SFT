"""
MCP Tools - SQLite DB 직접 조회
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import calendar

# DB 경로
DB_PATH = Path(__file__).parent.parent / "demand_data" / "demand.db"


def get_db_connection():
    """SQLite 연결"""
    return sqlite3.connect(DB_PATH)


class DemandTools:
    """전력수요 데이터 조회 도구 (SQLite 직접 조회)"""

    def get_demand_summary(self, year: int, month: int) -> Dict[str, Any]:
        """
        월간 전력수요 요약 조회

        Returns:
            - max_demand: 최대부하 (MW)
            - avg_demand: 평균부하 (MW)
            - min_demand: 최소부하 (MW)
            - yoy_change: 전년비 변동률 (%)
        """
        conn = get_db_connection()
        cur = conn.cursor()

        # 해당 월 데이터 조회
        cur.execute("""
            SELECT
                MAX(demand_max) as max_demand,
                AVG(demand_mean) as avg_demand,
                MIN(demand_min) as min_demand,
                COUNT(*) as hours
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
        """, (str(year), f"{month:02d}"))

        row = cur.fetchone()

        if not row or row[0] is None:
            conn.close()
            return {"error": f"{year}년 {month}월 데이터 없음"}

        max_demand, avg_demand, min_demand, hours = row

        # 전년 동월 데이터 (YoY 계산용)
        cur.execute("""
            SELECT MAX(demand_max)
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
        """, (str(year - 1), f"{month:02d}"))

        prev_row = cur.fetchone()
        prev_max = prev_row[0] if prev_row and prev_row[0] else None

        yoy_change = None
        if prev_max:
            yoy_change = round((max_demand - prev_max) / prev_max * 100, 2)

        conn.close()

        return {
            "year": year,
            "month": month,
            "max_demand": round(max_demand, 0),
            "avg_demand": round(avg_demand, 0),
            "min_demand": round(min_demand, 0),
            "hours": hours,
            "yoy_change": yoy_change,
        }

    def get_weekly_demand(self, year: int, month: int) -> List[Dict[str, Any]]:
        """주별 전력수요 조회 (날짜 범위 포함)"""
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                strftime('%W', timestamp) as week,
                MAX(demand_max) as max_demand,
                AVG(demand_mean) as avg_demand,
                MIN(demand_min) as min_demand,
                MIN(date(timestamp)) as start_date,
                MAX(date(timestamp)) as end_date
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
            GROUP BY week
            ORDER BY week
        """, (str(year), f"{month:02d}"))

        rows = cur.fetchall()
        conn.close()

        result = []
        for i, row in enumerate(rows, 1):
            start_date = row[4] if row[4] else ""
            end_date = row[5] if row[5] else ""

            # 날짜 범위 문자열 생성 (예: "(7/1~7/7)")
            date_range = ""
            if start_date and end_date:
                try:
                    s_date = datetime.strptime(start_date, "%Y-%m-%d")
                    e_date = datetime.strptime(end_date, "%Y-%m-%d")
                    date_range = f"({s_date.month}/{s_date.day}~{e_date.month}/{e_date.day})"
                except ValueError:
                    date_range = ""

            result.append({
                "week": i,
                "week_label": f"{i}주",
                "max_demand": round(row[1], 0) if row[1] else None,
                "avg_demand": round(row[2], 0) if row[2] else None,
                "min_demand": round(row[3], 0) if row[3] else None,
                "start_date": start_date,
                "end_date": end_date,
                "date_range": date_range,
            })

        return result

    def get_peak_load(self, year: int, month: int) -> Dict[str, Any]:
        """최대부하 상세 정보 조회"""
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT timestamp, demand_max, demand_mean
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
            ORDER BY demand_max DESC
            LIMIT 1
        """, (str(year), f"{month:02d}"))

        row = cur.fetchone()
        conn.close()

        if not row:
            return {"error": f"{year}년 {month}월 데이터 없음"}

        # 다양한 timestamp 형식 지원
        try:
            peak_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                peak_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                peak_time = datetime.strptime(row[0], "%Y-%m-%d")

        return {
            "peak_datetime": row[0],
            "peak_date": peak_time.strftime("%Y-%m-%d"),
            "peak_hour": peak_time.hour,
            "peak_demand": round(row[1], 0),
            "weekday": peak_time.strftime("%A"),
        }

    def get_historical_data(self, month: int, years: int = 5, target_year: int = None) -> List[Dict[str, Any]]:
        """과거 N년간 동월 데이터 조회 (YoY 증감률 포함)

        Args:
            month: 조회할 월
            years: 조회할 연도 수
            target_year: 기준 연도 (이 연도까지의 데이터만 조회, None이면 전체)
        """
        conn = get_db_connection()
        cur = conn.cursor()

        # 충분한 데이터를 가져와서 YoY 계산 가능하도록 (years+1개)
        if target_year:
            cur.execute("""
                SELECT
                    strftime('%Y', timestamp) as year,
                    MAX(demand_max) as max_demand,
                    AVG(demand_mean) as avg_demand
                FROM demand_1hour
                WHERE strftime('%m', timestamp) = ?
                  AND CAST(strftime('%Y', timestamp) AS INTEGER) <= ?
                GROUP BY year
                ORDER BY year DESC
                LIMIT ?
            """, (f"{month:02d}", target_year, years + 1))
        else:
            cur.execute("""
                SELECT
                    strftime('%Y', timestamp) as year,
                    MAX(demand_max) as max_demand,
                    AVG(demand_mean) as avg_demand
                FROM demand_1hour
                WHERE strftime('%m', timestamp) = ?
                GROUP BY year
                ORDER BY year DESC
                LIMIT ?
            """, (f"{month:02d}", years + 1))

        rows = cur.fetchall()
        conn.close()

        # 연도별 데이터를 딕셔너리로 변환
        year_data = {}
        for row in rows:
            year_data[int(row[0])] = {
                "max_demand": round(row[1], 0) if row[1] else None,
                "avg_demand": round(row[2], 0) if row[2] else None,
            }

        # 최근 years개 연도에 대해 YoY 계산
        result = []
        sorted_years = sorted(year_data.keys(), reverse=True)[:years]

        for yr in sorted_years:
            data = year_data[yr]
            prev_yr = yr - 1

            # 전년 데이터가 있으면 YoY 계산
            max_yoy = None
            avg_yoy = None
            if prev_yr in year_data:
                prev_data = year_data[prev_yr]
                if data["max_demand"] and prev_data["max_demand"]:
                    max_yoy = round((data["max_demand"] - prev_data["max_demand"]) / prev_data["max_demand"] * 100, 1)
                if data["avg_demand"] and prev_data["avg_demand"]:
                    avg_yoy = round((data["avg_demand"] - prev_data["avg_demand"]) / prev_data["avg_demand"] * 100, 1)

            result.append({
                "year": yr,
                "max_demand": data["max_demand"],
                "avg_demand": data["avg_demand"],
                "max_yoy": max_yoy,
                "avg_yoy": avg_yoy,
            })

        return result

    def get_daily_pattern(self, year: int, month: int, day_type: int = 0) -> List[Dict[str, Any]]:
        """
        시간대별 수요 패턴 조회

        Args:
            day_type: 0=평일, 1=주말, 2=공휴일
        """
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                strftime('%H', timestamp) as hour,
                AVG(demand_mean) as avg_demand,
                MAX(demand_max) as max_demand
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
              AND day_type = ?
            GROUP BY hour
            ORDER BY hour
        """, (str(year), f"{month:02d}", day_type))

        rows = cur.fetchall()
        conn.close()

        return [
            {
                "hour": int(row[0]),
                "avg_demand": round(row[1], 0) if row[1] else None,
                "max_demand": round(row[2], 0) if row[2] else None,
            }
            for row in rows
        ]


class WeatherTools:
    """기상 데이터 조회 도구 (SQLite 직접 조회)"""

    def get_weather_summary(self, year: int, month: int) -> Dict[str, Any]:
        """월간 기상 요약 조회"""
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                AVG(temperature_mean) as temp_avg,
                MAX(temperature_max) as temp_max,
                MIN(temperature_min) as temp_min,
                AVG(humidity_mean) as humidity_avg,
                COUNT(*) as hours
            FROM weather_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
        """, (str(year), f"{month:02d}"))

        row = cur.fetchone()
        conn.close()

        if not row or row[0] is None:
            return {"error": f"{year}년 {month}월 기상 데이터 없음"}

        return {
            "year": year,
            "month": month,
            "temperature_avg": round(row[0], 1),
            "temperature_max": round(row[1], 1),
            "temperature_min": round(row[2], 1),
            "humidity_avg": round(row[3], 1),
            "hours": row[4],
        }

    def get_daily_weather(self, year: int, month: int) -> List[Dict[str, Any]]:
        """일별 기상 데이터 조회"""
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                date(timestamp) as date,
                AVG(temperature_mean) as temp_avg,
                MAX(temperature_max) as temp_max,
                MIN(temperature_min) as temp_min,
                AVG(humidity_mean) as humidity_avg
            FROM weather_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND strftime('%m', timestamp) = ?
            GROUP BY date
            ORDER BY date
        """, (str(year), f"{month:02d}"))

        rows = cur.fetchall()
        conn.close()

        return [
            {
                "date": row[0],
                "temperature_avg": round(row[1], 1) if row[1] else None,
                "temperature_max": round(row[2], 1) if row[2] else None,
                "temperature_min": round(row[3], 1) if row[3] else None,
                "humidity_avg": round(row[4], 1) if row[4] else None,
            }
            for row in rows
        ]


class CombinedTools:
    """통합 조회 도구"""

    def __init__(self):
        self.demand_tools = DemandTools()
        self.weather_tools = WeatherTools()

    def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        """보고서 생성용 전체 데이터 조회"""
        summary = self.demand_tools.get_demand_summary(year, month)
        weekly = self.demand_tools.get_weekly_demand(year, month)
        peak = self.demand_tools.get_peak_load(year, month)
        historical = self.demand_tools.get_historical_data(month, years=5, target_year=year)
        weekday_pattern = self.demand_tools.get_daily_pattern(year, month, day_type=0)
        weekend_pattern = self.demand_tools.get_daily_pattern(year, month, day_type=1)
        weather = self.weather_tools.get_weather_summary(year, month)

        return {
            "summary": summary,
            "weekly_demand": weekly,
            "peak_load": peak,
            "historical": historical,
            "weekday_pattern": weekday_pattern,
            "weekend_pattern": weekend_pattern,
            "weather": weather,
        }

    def get_weather_summary(self, year: int, month: int) -> Dict[str, Any]:
        return self.weather_tools.get_weather_summary(year, month)

    def get_demand_summary(self, year: int, month: int) -> Dict[str, Any]:
        return self.demand_tools.get_demand_summary(year, month)

    def get_weekly_demand(self, year: int, month: int) -> List[Dict[str, Any]]:
        return self.demand_tools.get_weekly_demand(year, month)

    def get_peak_load(self, year: int, month: int) -> Dict[str, Any]:
        return self.demand_tools.get_peak_load(year, month)

    def get_historical_demand(self, month: int, years: int = 5, target_year: int = None) -> List[Dict[str, Any]]:
        return self.demand_tools.get_historical_data(month, years, target_year=target_year)


class ChartTools:
    """차트 생성 도구 (MCP용)"""

    def __init__(self):
        self.demand_tools = DemandTools()
        self.output_dir = Path(__file__).parent.parent / "reports"

    def generate_weekly_chart(self, year: int, month: int) -> Dict[str, Any]:
        """
        주별 전력수요 추이 라인 차트 생성
        
        Args:
            year: 연도
            month: 월 (1-12)
            
        Returns:
            - success: 성공 여부
            - filepath: 생성된 차트 파일 경로
            - error: 에러 메시지 (실패 시)
        """
        try:
            import matplotlib.pyplot as plt
            
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 데이터 조회
            weekly_data = self.demand_tools.get_weekly_demand(year, month)
            
            if not weekly_data:
                return {"success": False, "error": f"{year}년 {month}월 데이터 없음"}
            
            # 차트 저장 디렉토리 생성
            charts_dir = self.output_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # 데이터 추출
            weeks = [f"W{d['week']}" for d in weekly_data]
            max_demands = [d['max_demand'] / 10000 if d['max_demand'] else 0 for d in weekly_data]
            avg_demands = [d['avg_demand'] / 10000 if d['avg_demand'] else 0 for d in weekly_data]
            min_demands = [d['min_demand'] / 10000 if d['min_demand'] else 0 for d in weekly_data]
            
            # 차트 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 라인 플롯
            ax.plot(weeks, max_demands, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Max (10MW)')
            ax.plot(weeks, avg_demands, 's-', color='#3498db', linewidth=2, markersize=8, label='Avg (10MW)')
            ax.plot(weeks, min_demands, '^-', color='#2ecc71', linewidth=2, markersize=8, label='Min (10MW)')
            
            # 최대/최소 영역 채우기
            ax.fill_between(weeks, min_demands, max_demands, alpha=0.1, color='#3498db')
            
            # 데이터 레이블
            for w, max_d, avg_d in zip(weeks, max_demands, avg_demands):
                ax.annotate(f'{max_d:.1f}', (w, max_d), textcoords="offset points", 
                           xytext=(0, 10), ha='center', fontsize=9, color='#e74c3c')
                ax.annotate(f'{avg_d:.1f}', (w, avg_d), textcoords="offset points", 
                           xytext=(0, -15), ha='center', fontsize=9, color='#3498db')
            
            # 스타일링
            ax.set_xlabel('Week', fontsize=12)
            ax.set_ylabel('Power Demand (10MW)', fontsize=12)
            ax.set_title(f'Weekly Power Demand Trend - {year}/{month:02d}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(bottom=0)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            
            # 파일 저장
            filename = f"weekly_demand_{year}_{month:02d}.png"
            filepath = charts_dir / filename
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "filepath": str(filepath),
                "filename": filename,
                "weeks": len(weekly_data),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
