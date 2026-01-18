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

    def __init__(self):
        self._forecast_tools = None

    @property
    def forecast_tools(self):
        """ForecastTools 지연 로딩 (모델 로드 비용 절감)"""
        if self._forecast_tools is None:
            self._forecast_tools = ForecastTools()
        return self._forecast_tools

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

        # 기준 연도 데이터가 없으면 월별 예측값으로 보완
        if target_year and target_year not in year_data:
            forecast = self.forecast_tools.forecast_monthly_demand(target_year, month)
            if forecast.get("avg_demand") or forecast.get("max_demand"):
                year_data[target_year] = {
                    "max_demand": forecast.get("max_demand"),
                    "avg_demand": forecast.get("avg_demand"),
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

    def get_monthly_demand_by_year(self, year: int, up_to_month: int = 12) -> List[Dict[str, Any]]:
        """특정 연도의 월별 평균 수요 조회

        Args:
            year: 연도
            up_to_month: 이 월까지만 조회 (1-12)

        Returns:
            월별 평균/최대 수요 리스트
        """
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                CAST(strftime('%m', timestamp) AS INTEGER) as month,
                AVG(demand_mean) as avg_demand,
                MAX(demand_max) as max_demand
            FROM demand_1hour
            WHERE strftime('%Y', timestamp) = ?
              AND CAST(strftime('%m', timestamp) AS INTEGER) <= ?
            GROUP BY month
            ORDER BY month
        """, (str(year), up_to_month))

        rows = cur.fetchall()
        conn.close()

        return [
            {
                "month": row[0],
                "avg_demand": round(row[1], 0) if row[1] else None,
                "max_demand": round(row[2], 0) if row[2] else None,
            }
            for row in rows
        ]

    def get_yearly_monthly_demand(self, target_year: int, target_month: int, years: int = 5) -> Dict[str, List]:
        """최근 N년간 월별 평균 수요 조회 (차트용)

        Args:
            target_year: 기준 연도
            target_month: 기준 월 (이 연도는 이 월까지만 조회)
            years: 조회할 연도 수

        Returns:
            연도별 월별 수요 데이터
        """
        result = {}

        for i in range(years):
            year = target_year - i
            # 기준 연도는 target_month까지만, 나머지는 12월까지
            up_to_month = target_month if year == target_year else 12
            monthly_data = self.get_monthly_demand_by_year(year, up_to_month)

            # 기준 연도 target_month 데이터가 없으면 예측값 추가
            if year == target_year:
                has_target_month = any(d.get("month") == target_month for d in monthly_data)
                if not has_target_month:
                    forecast = self.forecast_tools.forecast_monthly_demand(target_year, target_month)
                    if forecast.get("avg_demand") or forecast.get("max_demand"):
                        monthly_data.append({
                            "month": target_month,
                            "avg_demand": forecast.get("avg_demand"),
                            "max_demand": forecast.get("max_demand"),
                        })
                        monthly_data = sorted(monthly_data, key=lambda x: x.get("month", 0))

            if monthly_data:
                result[year] = monthly_data

        return result


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

    def get_historical_weather(self, month: int, years: int = 5, target_year: int = None) -> Dict[str, Any]:
        """과거 N년간 동월 기상 데이터 조회

        Args:
            month: 조회할 월
            years: 조회할 연도 수
            target_year: 기준 연도 (이 연도 이전 데이터만 조회)

        Returns:
            - years_data: 연도별 기상 데이터 리스트
            - avg_temperature: 과거 N년 평균 기온
            - avg_humidity: 과거 N년 평균 습도
        """
        conn = get_db_connection()
        cur = conn.cursor()

        if target_year:
            cur.execute("""
                SELECT
                    strftime('%Y', timestamp) as year,
                    AVG(temperature_mean) as temp_avg,
                    MAX(temperature_max) as temp_max,
                    MIN(temperature_min) as temp_min,
                    AVG(humidity_mean) as humidity_avg
                FROM weather_1hour
                WHERE strftime('%m', timestamp) = ?
                  AND CAST(strftime('%Y', timestamp) AS INTEGER) < ?
                GROUP BY year
                ORDER BY year DESC
                LIMIT ?
            """, (f"{month:02d}", target_year, years))
        else:
            cur.execute("""
                SELECT
                    strftime('%Y', timestamp) as year,
                    AVG(temperature_mean) as temp_avg,
                    MAX(temperature_max) as temp_max,
                    MIN(temperature_min) as temp_min,
                    AVG(humidity_mean) as humidity_avg
                FROM weather_1hour
                WHERE strftime('%m', timestamp) = ?
                GROUP BY year
                ORDER BY year DESC
                LIMIT ?
            """, (f"{month:02d}", years))

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return {"error": f"{month}월 과거 기상 데이터 없음"}

        years_data = []
        total_temp = 0
        total_humidity = 0
        valid_count = 0

        for row in rows:
            temp_avg = round(row[1], 1) if row[1] else None
            humidity_avg = round(row[4], 1) if row[4] else None

            years_data.append({
                "year": int(row[0]),
                "temperature_avg": temp_avg,
                "temperature_max": round(row[2], 1) if row[2] else None,
                "temperature_min": round(row[3], 1) if row[3] else None,
                "humidity_avg": humidity_avg,
            })

            if temp_avg is not None:
                total_temp += temp_avg
                valid_count += 1
            if humidity_avg is not None:
                total_humidity += humidity_avg

        return {
            "month": month,
            "years_count": len(years_data),
            "years_data": years_data,
            "avg_temperature": round(total_temp / valid_count, 1) if valid_count > 0 else None,
            "avg_humidity": round(total_humidity / valid_count, 1) if valid_count > 0 else None,
        }


class CombinedTools:
    """통합 조회 도구"""

    def __init__(self):
        self.demand_tools = DemandTools()
        self.weather_tools = WeatherTools()
        self._forecast_tools = None  # 지연 로딩

    @property
    def forecast_tools(self):
        """ForecastTools 지연 로딩 (GPU 메모리 절약)"""
        if self._forecast_tools is None:
            self._forecast_tools = ForecastTools()
        return self._forecast_tools

    def get_report_data(self, year: int, month: int) -> Dict[str, Any]:
        """보고서 생성용 전체 데이터 조회"""
        summary = self.demand_tools.get_demand_summary(year, month)
        weekly = self.demand_tools.get_weekly_demand(year, month)
        peak = self.demand_tools.get_peak_load(year, month)
        historical = self.demand_tools.get_historical_data(month, years=5, target_year=year)
        weekday_pattern = self.demand_tools.get_daily_pattern(year, month, day_type=0)
        weekend_pattern = self.demand_tools.get_daily_pattern(year, month, day_type=1)
        weather = self.weather_tools.get_weather_summary(year, month)
        # 과거 동월 기상 데이터 (기상전망 작성용)
        historical_weather = self.weather_tools.get_historical_weather(month, years=5, target_year=year)
        monthly_forecast = self.forecast_tools.forecast_monthly_demand(year, month)

        return {
            "summary": summary,
            "weekly_demand": weekly,
            "peak_load": peak,
            "historical": historical,
            "weekday_pattern": weekday_pattern,
            "weekend_pattern": weekend_pattern,
            "weather": weather,
            "historical_weather": historical_weather,
            "monthly_forecast": monthly_forecast,
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

    def get_yearly_monthly_demand(self, target_year: int, target_month: int, years: int = 5) -> Dict[str, List]:
        return self.demand_tools.get_yearly_monthly_demand(target_year, target_month, years)

    def forecast_weekly_demand(self, year: int, month: int, model: str = "lstm", include_next_month: bool = False) -> Dict[str, Any]:
        """주차별 전력수요 예측

        Args:
            year: 예측 대상 연도
            month: 예측 대상 월
            model: 사용할 모델 ("arima", "holt_winters", "lstm", "ensemble")
            include_next_month: True면 다음 달까지 예측 (LSTM 8주 모델 사용)
        """
        return self.forecast_tools.forecast_weekly_demand(year, month, model, include_next_month)


class ForecastTools:
    """주차별 전력수요 예측 도구 (ARIMA, Holt-Winters, LSTM)"""

    def __init__(self):
        self.model_dir = Path(__file__).parent.parent.parent  # /root/De-Qwen-SFT
        self.scaler_x_4 = None
        self.scaler_y_4 = None
        self.scaler_x_8 = None
        self.scaler_y_8 = None
        self.scaler_x_monthly_mean = None
        self.scaler_y_monthly_mean = None
        self.scaler_x_monthly_peak = None
        self.scaler_y_monthly_peak = None
        # 4주/8주 모델 파라미터 분리
        self.lstm_params_4 = None
        self.lstm_params_8 = None
        self.monthly_models = {"mean": [], "peak": []}
        self._load_lstm_models()

    def _load_lstm_models(self):
        """LSTM 모델 및 스케일러 로드 (주차/월차 예측)"""
        try:
            import torch
            import pickle

            # ===== 주차별 LSTM =====
            model_path_4 = self.model_dir / "best_direct_lstm_full_weekly_max_h4_win12.pth"
            model_path_8 = self.model_dir / "best_direct_lstm_full_weekly_max_h8_win12.pth"
            scaler_path_4 = self.model_dir / "scalers_weekly_max_h4_win12.pkl"
            scaler_path_8 = self.model_dir / "scalers_weekly_max_h8_win12.pkl"

            if scaler_path_4.exists():
                with open(scaler_path_4, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_x_4 = scalers.get('scaler_x')
                    self.scaler_y_4 = scalers.get('scaler_y')
            else:
                print(f"[WARN] 스케일러 파일 없음: {scaler_path_4}")

            if scaler_path_8.exists():
                with open(scaler_path_8, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_x_8 = scalers.get('scaler_x')
                    self.scaler_y_8 = scalers.get('scaler_y')
            else:
                print(f"[WARN] 스케일러 파일 없음: {scaler_path_8}")

            if model_path_4.exists():
                checkpoint_4 = torch.load(model_path_4, map_location='cpu', weights_only=False)
                params_4 = checkpoint_4.get('params', {})
                params_4['model_path'] = model_path_4
                params_4['feature_cols'] = checkpoint_4.get('feature_cols', ["demand_max", "is_holiday", "day_type"])
                params_4['target_col'] = checkpoint_4.get('target_col', "demand_max")
                params_4['n_input'] = checkpoint_4.get('n_input', 12)
                params_4['n_output'] = checkpoint_4.get('n_output', 4)
                self.lstm_params_4 = params_4
                print(f"[INFO] LSTM 4주 모델 로드 완료")
            else:
                print(f"[WARN] LSTM 4주 모델 파일 없음: {model_path_4}")

            if model_path_8.exists():
                checkpoint_8 = torch.load(model_path_8, map_location='cpu', weights_only=False)
                params_8 = checkpoint_8.get('params', {})
                params_8['model_path'] = model_path_8
                params_8['feature_cols'] = checkpoint_8.get('feature_cols', ["demand_max", "is_holiday", "day_type"])
                params_8['target_col'] = checkpoint_8.get('target_col', "demand_max")
                params_8['n_input'] = checkpoint_8.get('n_input', 12)
                params_8['n_output'] = checkpoint_8.get('n_output', 8)
                self.lstm_params_8 = params_8
                print(f"[INFO] LSTM 8주 모델 로드 완료")
            else:
                print(f"[WARN] LSTM 8주 모델 파일 없음: {model_path_8}")

            # ===== 월차별 LSTM =====
            mean_scaler_path = self.model_dir / "scalers_monthly_mean.pkl"
            peak_scaler_path = self.model_dir / "scalers_monthly_peak.pkl"

            if mean_scaler_path.exists():
                with open(mean_scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_x_monthly_mean = scalers.get('scaler_x')
                    self.scaler_y_monthly_mean = scalers.get('scaler_y')
            else:
                print(f"[WARN] 스케일러 파일 없음: {mean_scaler_path}")

            if peak_scaler_path.exists():
                with open(peak_scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.scaler_x_monthly_peak = scalers.get('scaler_x')
                    self.scaler_y_monthly_peak = scalers.get('scaler_y')
            else:
                print(f"[WARN] 스케일러 파일 없음: {peak_scaler_path}")

            self.monthly_models = {
                "mean": self._load_many2one_models(
                    [self.model_dir / "best_many2one_lstm_state_monthly_mean_win12.pth"],
                    default_feature_cols=["demand_mean", "is_holiday", "day_type"],
                    default_target_col="demand_mean",
                ),
                "peak": self._load_many2one_models(
                    [
                        self.model_dir / "best_many2one_lstm_state_monthly_peak_win12.pth",
                    ],
                    default_feature_cols=["demand_max", "is_holiday", "day_type"],
                    default_target_col="demand_max",
                ),
            }

        except Exception as e:
            print(f"[WARN] LSTM 모델 로드 실패: {e}")

    def _load_many2one_models(self, model_paths: List[Path], default_feature_cols: List[str], default_target_col: str):
        try:
            import torch
        except Exception:
            return []

        models = []
        for model_path in model_paths:
            if not model_path.exists():
                print(f"[WARN] 월별 LSTM 모델 파일 없음: {model_path}")
                continue

            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'params' in checkpoint:
                params = checkpoint.get('params', {})
                params['model_path'] = model_path
                params['model_state'] = checkpoint.get('model_state')
                params['n_input'] = checkpoint.get('n_input', 12)
                params['feature_cols'] = checkpoint.get('feature_cols', default_feature_cols)
                params['target_col'] = checkpoint.get('target_col', default_target_col)
                if 'use_attention' not in params:
                    params['use_attention'] = any(
                        k.startswith('attention.') for k in (params.get('model_state') or {}).keys()
                    )
                params['input_size'] = len(params['feature_cols'])
                models.append(params)
            else:
                state_dict = checkpoint
                if not isinstance(state_dict, dict):
                    continue

                hidden_size = state_dict['encoder.weight_ih_l0'].shape[0] // 4
                input_size = state_dict['encoder.weight_ih_l0'].shape[1]
                num_layers = 2 if 'encoder.weight_ih_l1' in state_dict else 1
                use_attention = any(k.startswith('attention.') for k in state_dict.keys())

                models.append({
                    "model_path": model_path,
                    "model_state": state_dict,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "use_attention": use_attention,
                    "n_input": 12,
                    "feature_cols": default_feature_cols,
                    "target_col": default_target_col,
                    "input_size": input_size,
                })

        return models

    def _get_weekly_data(self, up_to_year: int, up_to_month: int):
        """주차별 데이터 준비 (예측을 위한 과거 데이터)"""
        import pandas as pd
        conn = get_db_connection()

        # up_to_year, up_to_month 이전까지의 모든 데이터 조회
        # 주차별 모델은 demand_max 기반이며, 필요 컬럼을 모두 조회
        query = """
            SELECT timestamp, demand_mean, demand_max, is_holiday, day_type
            FROM demand_1hour
            WHERE timestamp < ?
            ORDER BY timestamp
        """
        # 해당 월의 첫날
        end_date = f"{up_to_year}-{up_to_month:02d}-01"

        df = pd.read_sql(query, conn, params=[end_date])
        conn.close()

        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "demand_mean", "demand_max", "is_holiday", "day_type"]]

        # 주차별 리샘플링
        df_weekly = df.set_index("timestamp").resample('W').agg({
            'demand_mean': 'mean',
            'demand_max': 'max',
            'is_holiday': 'mean',
            'day_type': 'mean'
        })
        df_weekly = df_weekly.dropna()

        return df_weekly

    def _get_monthly_data(self, up_to_year: int, up_to_month: int):
        """월별 데이터 준비 (예측을 위한 과거 데이터)"""
        import pandas as pd
        conn = get_db_connection()

        query = """
            SELECT timestamp, demand_mean, demand_max, is_holiday, day_type
            FROM demand_1hour
            WHERE timestamp < ?
            ORDER BY timestamp
        """
        end_date = f"{up_to_year}-{up_to_month:02d}-01"

        df = pd.read_sql(query, conn, params=[end_date])
        conn.close()

        if df.empty:
            return None

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "demand_mean", "demand_max", "is_holiday", "day_type"]]

        df_monthly = df.set_index("timestamp").resample('M').agg({
            'demand_mean': 'mean',
            'demand_max': 'max',
            'is_holiday': 'mean',
            'day_type': 'mean'
        })
        df_monthly = df_monthly.dropna()

        return df_monthly

    def _get_weeks_in_month(self, year: int, month: int) -> List[Dict]:
        """해당 월의 주차 정보 계산"""
        import calendar
        from datetime import date, timedelta

        first_day = date(year, month, 1)
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        weeks = []
        week_num = 1
        current = first_day

        while current <= last_day:
            week_start = current
            # 주의 끝 (일요일) 또는 월말
            days_until_sunday = (6 - current.weekday()) % 7
            week_end = min(current + timedelta(days=days_until_sunday), last_day)

            weeks.append({
                "week": week_num,
                "start_date": week_start,
                "end_date": week_end,
                "date_range": f"({week_start.month}/{week_start.day}~{week_end.month}/{week_end.day})"
            })

            week_num += 1
            current = week_end + timedelta(days=1)

        return weeks

    def _forecast_many2one(self, df_monthly, model_info: Dict[str, Any], scaler_x, scaler_y):
        try:
            import torch
            import torch.nn as nn
        except Exception:
            return None

        if scaler_x is None or scaler_y is None:
            return None

        feature_cols = model_info.get("feature_cols", ["demand_mean", "is_holiday", "day_type"])
        n_input = model_info.get("n_input", 12)
        if df_monthly is None or len(df_monthly) < n_input:
            return None

        data_x = df_monthly[feature_cols]
        data_x_scaled = scaler_x.transform(data_x)
        last_sequence = data_x_scaled[-n_input:]

        class Many2OneLSTM(nn.Module):
            def __init__(self, input_size=3, hidden_size=64, num_layers=1):
                super(Many2OneLSTM, self).__init__()
                self.encoder = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 1)
                )

            def forward(self, x):
                _, (hidden, _) = self.encoder(x)
                last_hidden = hidden[-1]
                out = self.fc(last_hidden)
                return out.unsqueeze(1)

        class Many2OneLSTMWithAttention(nn.Module):
            def __init__(self, input_size=3, hidden_size=64, num_layers=1):
                super(Many2OneLSTMWithAttention, self).__init__()
                self.encoder = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0
                )
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_size // 2, 1)
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, 1)
                )

            def forward(self, x):
                encoder_output, _ = self.encoder(x)
                attn_scores = self.attention(encoder_output)
                attn_weights = torch.softmax(attn_scores, dim=1)
                context = torch.sum(encoder_output * attn_weights, dim=1)
                out = self.fc(context)
                return out.unsqueeze(1)

        use_attention = model_info.get("use_attention", False)
        hidden_size = model_info.get("hidden_size", 64)
        num_layers = model_info.get("num_layers", 1)
        input_size = model_info.get("input_size", len(feature_cols))

        if use_attention:
            model = Many2OneLSTMWithAttention(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        else:
            model = Many2OneLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )

        model_state = model_info.get("model_state")
        if model_state is None:
            return None

        model.load_state_dict(model_state)
        model.eval()

        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()

        pred_value = scaler_y.inverse_transform(pred_scaled.reshape(1, 1))[0, 0]
        return float(pred_value)

    def _forecast_holt_winters_monthly(self, df_monthly, target_col: str):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except Exception:
            return None

        if df_monthly is None or target_col not in df_monthly.columns:
            return None

        series = df_monthly[target_col]
        if len(series) < 24:
            return None

        try:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=12
            ).fit()
            forecast = model.forecast(steps=1)
            return float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
        except Exception:
            return None

    def _forecast_arima_monthly(self, df_monthly, target_col: str):
        if df_monthly is None or target_col not in df_monthly.columns:
            return None

        series = df_monthly[target_col]
        if len(series) < 24:
            return None

        try:
            from pmdarima import auto_arima
            model = auto_arima(series, seasonal=False, error_action='ignore', suppress_warnings=True)
            forecast = model.predict(n_periods=1)
            return float(forecast[-1])
        except Exception:
            pass

        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=(1, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            return float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
        except Exception:
            return None

    def forecast_monthly_demand(self, year: int, month: int) -> Dict[str, Any]:
        """월별 평균/최대 수요 예측 (LSTM + Holt-Winters + ARIMA 평균)

        Returns:
            - year, month: 대상 연월
            - avg_demand: 평균부하 예측값 (3개 모델 평균)
            - max_demand: 최대부하 예측값 (3개 모델 평균)
            - model_predictions: 개별 모델 예측값 딕셔너리
              - mean: {lstm, holt_winters, arima}
              - peak: {lstm, holt_winters, arima}
        """
        df_monthly = self._get_monthly_data(year, month)
        if df_monthly is None:
            return {}

        # 개별 모델 예측값 저장
        model_preds = {
            "mean": {"lstm": None, "holt_winters": None, "arima": None},
            "peak": {"lstm": None, "holt_winters": None, "arima": None},
        }

        mean_preds = []
        # LSTM 모델
        for model_info in self.monthly_models.get("mean", []):
            pred = self._forecast_many2one(
                df_monthly,
                model_info,
                self.scaler_x_monthly_mean,
                self.scaler_y_monthly_mean
            )
            if pred is not None:
                mean_preds.append(pred)
                model_preds["mean"]["lstm"] = round(pred, 0)

        # Holt-Winters 모델
        mean_hw = self._forecast_holt_winters_monthly(df_monthly, "demand_mean")
        if mean_hw is not None:
            mean_preds.append(mean_hw)
            model_preds["mean"]["holt_winters"] = round(mean_hw, 0)

        # ARIMA 모델
        mean_arima = self._forecast_arima_monthly(df_monthly, "demand_mean")
        if mean_arima is not None:
            mean_preds.append(mean_arima)
            model_preds["mean"]["arima"] = round(mean_arima, 0)

        peak_preds = []
        # LSTM 모델
        for model_info in self.monthly_models.get("peak", []):
            pred = self._forecast_many2one(
                df_monthly,
                model_info,
                self.scaler_x_monthly_peak,
                self.scaler_y_monthly_peak
            )
            if pred is not None:
                peak_preds.append(pred)
                model_preds["peak"]["lstm"] = round(pred, 0)

        # Holt-Winters 모델
        peak_hw = self._forecast_holt_winters_monthly(df_monthly, "demand_max")
        if peak_hw is not None:
            peak_preds.append(peak_hw)
            model_preds["peak"]["holt_winters"] = round(peak_hw, 0)

        # ARIMA 모델
        peak_arima = self._forecast_arima_monthly(df_monthly, "demand_max")
        if peak_arima is not None:
            peak_preds.append(peak_arima)
            model_preds["peak"]["arima"] = round(peak_arima, 0)

        mean_avg = round(sum(mean_preds) / len(mean_preds), 0) if mean_preds else None
        peak_avg = round(sum(peak_preds) / len(peak_preds), 0) if peak_preds else None

        return {
            "year": year,
            "month": month,
            "avg_demand": mean_avg,
            "max_demand": peak_avg,
            "mean_models": mean_preds,
            "peak_models": peak_preds,
            "model_predictions": model_preds,
        }

    def forecast_arima(self, year: int, month: int) -> List[Dict]:
        """ARIMA 모델로 주차별 예측"""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            df_weekly = self._get_weekly_data(year, month)
            if df_weekly is None or len(df_weekly) < 52:
                return []

            weeks = self._get_weeks_in_month(year, month)
            n_weeks = len(weeks)

            # ARIMA(1,1,0) 모델 학습
            model = ARIMA(df_weekly['demand_mean'], order=(1, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=n_weeks)

            results = []
            for i, (week_info, pred) in enumerate(zip(weeks, forecast)):
                results.append({
                    "week": week_info["week"],
                    "date_range": week_info["date_range"],
                    "max_demand": round(pred, 0),  # 주차별 평균 → 최대로 사용
                    "model": "ARIMA"
                })

            return results

        except Exception as e:
            print(f"[ERROR] ARIMA 예측 실패: {e}")
            return []

    def forecast_holt_winters(self, year: int, month: int) -> List[Dict]:
        """Holt-Winters 모델로 주차별 예측"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            df_weekly = self._get_weekly_data(year, month)
            if df_weekly is None or len(df_weekly) < 104:  # 최소 2년
                return []

            weeks = self._get_weeks_in_month(year, month)
            n_weeks = len(weeks)

            # Holt-Winters 모델 (가법 계절성, 52주 주기)
            model = ExponentialSmoothing(
                df_weekly['demand_mean'],
                trend='add',
                seasonal='add',
                seasonal_periods=52
            ).fit()
            forecast = model.forecast(steps=n_weeks)

            results = []
            for i, (week_info, pred) in enumerate(zip(weeks, forecast.values)):
                results.append({
                    "week": week_info["week"],
                    "date_range": week_info["date_range"],
                    "max_demand": round(pred, 0),
                    "model": "Holt-Winters"
                })

            return results

        except Exception as e:
            print(f"[ERROR] Holt-Winters 예측 실패: {e}")
            return []

    def forecast_lstm(self, year: int, month: int, include_next_month: bool = False) -> List[Dict]:
        """LSTM 모델로 주차별 예측

        Args:
            year: 예측 대상 연도
            month: 예측 대상 월
            include_next_month: True면 8주 모델로 다음 달까지 예측

        Returns:
            주차별 예측 결과 리스트
        """
        try:
            import torch
            import torch.nn as nn
            import numpy as np
            import pandas as pd

            # 4주/8주 모델 선택
            forecast_steps = 8 if include_next_month else 4
            params = self.lstm_params_8 if include_next_month else self.lstm_params_4
            scaler_x = self.scaler_x_8 if include_next_month else self.scaler_x_4
            scaler_y = self.scaler_y_8 if include_next_month else self.scaler_y_4

            if scaler_x is None or scaler_y is None or params is None:
                print(f"[WARN] LSTM {forecast_steps}주 모델이 로드되지 않음")
                return []

            forecast_steps = params.get('n_output', 8 if include_next_month else 4)

            df_weekly = self._get_weekly_data(year, month)
            n_input = params.get('n_input', 12)
            if df_weekly is None or len(df_weekly) < n_input:
                return []

            # 현재 월 주차 정보
            weeks = self._get_weeks_in_month(year, month)

            # 다음 달 정보 초기화
            next_year = year if month < 12 else year + 1
            next_month = month + 1 if month < 12 else 1

            # 다음 달 주차 정보 (include_next_month일 때)
            next_month_weeks = []
            if include_next_month:
                next_month_weeks = self._get_weeks_in_month(next_year, next_month)

            total_weeks = weeks + next_month_weeks
            n_weeks = len(total_weeks)

            # 데이터 스케일링 (DataFrame 형태 유지하여 feature names 경고 방지)
            feature_cols = params.get("feature_cols", ["demand_max", "is_holiday", "day_type"])
            data_x = df_weekly[feature_cols]
            data_x_scaled = scaler_x.transform(data_x)

            # 마지막 12주 시퀀스
            last_sequence = data_x_scaled[-n_input:]

            # LSTM 모델 정의
            class DirectLSTMWithAttention(nn.Module):
                def __init__(self, input_size=3, hidden_size=64, num_layers=1, forecast_steps=4):
                    super(DirectLSTMWithAttention, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.forecast_steps = forecast_steps
                    self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                          batch_first=True, dropout=0.1 if num_layers > 1 else 0)
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.Tanh(),
                        nn.Linear(hidden_size // 2, 1)
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, forecast_steps)
                    )

                def forward(self, x):
                    encoder_output, _ = self.encoder(x)
                    attn_scores = self.attention(encoder_output)
                    attn_weights = torch.softmax(attn_scores, dim=1)
                    context = torch.sum(encoder_output * attn_weights, dim=1)
                    output = self.fc(context)
                    return output.unsqueeze(-1)

            class DirectLSTM(nn.Module):
                def __init__(self, input_size=3, hidden_size=64, num_layers=1, forecast_steps=4):
                    super(DirectLSTM, self).__init__()
                    self.encoder = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                          batch_first=True, dropout=0.1 if num_layers > 1 else 0)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, forecast_steps)
                    )

                def forward(self, x):
                    _, (hidden, _) = self.encoder(x)
                    output = self.fc(hidden[-1])
                    return output.unsqueeze(-1)

            # 모델 인스턴스 생성
            if params.get('use_attention', False):
                model = DirectLSTMWithAttention(
                    input_size=3,
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 1),
                    forecast_steps=forecast_steps
                )
            else:
                model = DirectLSTM(
                    input_size=3,
                    hidden_size=params.get('hidden_size', 64),
                    num_layers=params.get('num_layers', 1),
                    forecast_steps=forecast_steps
                )

            # 모델 가중치 로드
            model_path = params.get('model_path')
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint.get('model_state', checkpoint))
            model.eval()

            # 예측 (한 번에 forecast_steps 주 예측)
            all_forecasts = []
            current_sequence = last_sequence.copy()

            while len(all_forecasts) < n_weeks:
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                with torch.no_grad():
                    pred_scaled = model(input_tensor).cpu().numpy()[0]

                # 역변환
                pred = scaler_y.inverse_transform(pred_scaled)
                all_forecasts.extend(pred.flatten().tolist())

                # 시퀀스 업데이트 (추가 예측 필요 시)
                for p in pred.flatten():
                    new_row_data = {col: 0 for col in feature_cols}
                    target_col = params.get("target_col")
                    if target_col in new_row_data:
                        new_row_data[target_col] = p
                    new_row = pd.DataFrame([new_row_data], columns=feature_cols)
                    new_row_scaled = scaler_x.transform(new_row)
                    current_sequence = np.vstack([current_sequence[1:], new_row_scaled])

            # 결과 구성
            results = []
            for i, week_info in enumerate(total_weeks):
                if i < len(all_forecasts):
                    # 다음 달 여부 표시
                    is_next_month = i >= len(weeks)
                    results.append({
                        "week": week_info["week"],
                        "date_range": week_info["date_range"],
                        "max_demand": round(all_forecasts[i], 0),
                        "model": "LSTM",
                        "month": next_month if is_next_month else month,
                        "year": next_year if is_next_month else year
                    })

            return results

        except Exception as e:
            import traceback
            print(f"[ERROR] LSTM 예측 실패: {e}")
            traceback.print_exc()
            return []

    def forecast_weekly_demand(self, year: int, month: int, model: str = "lstm", include_next_month: bool = False) -> Dict[str, Any]:
        """주차별 전력수요 예측 (메인 함수)

        Args:
            year: 예측 대상 연도
            month: 예측 대상 월
            model: 사용할 모델 ("arima", "holt_winters", "lstm", "ensemble")
            include_next_month: True면 다음 달까지 예측 (LSTM 8주 모델 사용)

        Returns:
            주차별 예측 결과
        """
        weeks = self._get_weeks_in_month(year, month)

        if model == "arima":
            forecasts = self.forecast_arima(year, month)
        elif model == "holt_winters":
            forecasts = self.forecast_holt_winters(year, month)
        elif model == "lstm":
            forecasts = self.forecast_lstm(year, month, include_next_month=include_next_month)
        elif model == "ensemble":
            # 앙상블: 3가지 모델의 평균
            arima_fc = self.forecast_arima(year, month)
            hw_fc = self.forecast_holt_winters(year, month)
            lstm_fc = self.forecast_lstm(year, month, include_next_month=include_next_month)

            forecasts = []
            for i, week_info in enumerate(weeks):
                values = []
                if i < len(arima_fc) and arima_fc[i].get("max_demand"):
                    values.append(arima_fc[i]["max_demand"])
                if i < len(hw_fc) and hw_fc[i].get("max_demand"):
                    values.append(hw_fc[i]["max_demand"])
                if i < len(lstm_fc) and lstm_fc[i].get("max_demand"):
                    values.append(lstm_fc[i]["max_demand"])

                if values:
                    avg_demand = sum(values) / len(values)
                    forecasts.append({
                        "week": week_info["week"],
                        "date_range": week_info["date_range"],
                        "max_demand": round(avg_demand, 0),
                        "model": "Ensemble"
                    })

            # 다음 달 예측은 LSTM만 사용 (앙상블에서)
            if include_next_month and len(lstm_fc) > len(weeks):
                for fc in lstm_fc[len(weeks):]:
                    forecasts.append(fc)
        else:
            forecasts = []

        return {
            "year": year,
            "month": month,
            "model": model,
            "forecasts": forecasts,
            "weeks_count": len(forecasts)
        }


class ChartTools:
    """차트 생성 도구 (MCP용)"""

    def __init__(self):
        self.demand_tools = DemandTools()
        self.output_dir = Path(__file__).parent.parent / "reports"

    def generate_weekly_chart(self, year: int, month: int, return_base64: bool = False) -> Dict[str, Any]:
        """
        주별 전력수요 추이 라인 차트 생성

        Args:
            year: 연도
            month: 월 (1-12)
            return_base64: True면 base64 인코딩된 이미지 반환 (MCP용)

        Returns:
            - success: 성공 여부
            - filepath: 생성된 차트 파일 경로
            - image_base64: base64 인코딩된 이미지 (return_base64=True일 때)
            - error: 에러 메시지 (실패 시)
        """
        try:
            import matplotlib.pyplot as plt
            import io
            import base64

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

            plt.tight_layout()

            # base64 반환 모드
            if return_base64:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                return {
                    "success": True,
                    "image_base64": image_base64,
                    "mime_type": "image/png",
                    "weeks": len(weekly_data),
                }

            # 파일 저장 모드
            filename = f"weekly_demand_{year}_{month:02d}.png"
            filepath = charts_dir / filename
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

    def generate_yearly_monthly_chart(
        self,
        target_year: int,
        target_month: int,
        years: int = 5,
        output_dir: str = "./reports",
        return_base64: bool = False,
    ) -> Dict[str, Any]:
        """연도별 월별 평균 수요 라인 차트 생성 (PNG)

        Args:
            target_year: 기준 연도
            target_month: 기준 월
            years: 표시할 연도 수
            output_dir: 출력 디렉토리
            return_base64: True면 base64 인코딩된 이미지 반환 (MCP용)
        """
        try:
            import matplotlib.pyplot as plt
            import io
            import base64

            yearly_data = self.demand_tools.get_yearly_monthly_demand(target_year, target_month, years)
            if not yearly_data:
                return {"success": False, "error": "연도별 월별 데이터 없음"}

            charts_dir = Path(output_dir) / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
            fig, ax = plt.subplots(figsize=(12, 6))
            sorted_years = sorted(yearly_data.keys())

            for idx, year in enumerate(sorted_years):
                monthly_data = yearly_data[year]
                months = [d['month'] for d in monthly_data]
                avg_demands = [d['avg_demand'] / 10000 if d['avg_demand'] else 0 for d in monthly_data]
                color = colors[idx % len(colors)]

                if year == target_year:
                    ax.plot(months, avg_demands, 'o-', color=color, linewidth=3, markersize=10,
                            label=f'{year}', zorder=10)
                    for m, d in zip(months, avg_demands):
                        ax.annotate(f'{d:.0f}', (m, d), textcoords="offset points",
                                    xytext=(0, 10), ha='center', fontsize=9, fontweight='bold', color=color)
                else:
                    ax.plot(months, avg_demands, 'o--', color=color, linewidth=1.5, markersize=6,
                            label=f'{year}', alpha=0.7)

            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.axvline(x=target_month, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Avg Power Demand (10MW)', fontsize=12)
            ax.set_title(f'Monthly Avg Power Demand by Year ({sorted_years[0]}-{sorted_years[-1]})',
                         fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', title='Year')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(bottom=0)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')

            plt.tight_layout()

            # base64 반환 모드
            if return_base64:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()

                return {
                    "success": True,
                    "image_base64": image_base64,
                    "mime_type": "image/png",
                }

            # 파일 저장 모드
            filename = f"yearly_monthly_demand_{target_year}_{target_month:02d}.png"
            filepath = charts_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            return {"success": True, "filepath": str(filepath), "filename": filename}

        except Exception as e:
            return {"success": False, "error": str(e)}
