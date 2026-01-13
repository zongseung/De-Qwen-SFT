"""
CSV 데이터를 SQLite DB에 수동으로 로드하는 스크립트

Usage:
    python load_csv_to_db.py                    # 5분 데이터 로드
    python load_csv_to_db.py --hourly           # 1시간 집계 데이터 로드
    python load_csv_to_db.py --all              # 둘 다 로드
"""

import os
import sys
from datetime import datetime

import pandas as pd
from workalendar.asia import SouthKorea

try:
    from .database import Demand5Min, Demand1Hour, Weather1Hour, init_db_sync, sync_engine, DB_PATH
except ImportError:
    from database import Demand5Min, Demand1Hour, Weather1Hour, init_db_sync, sync_engine, DB_PATH

# ========================================
# Constants
# ========================================
# CSV 경로 (상위 디렉토리)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_5MIN = os.path.join(_BASE_DIR, "Demand_Data_all.csv")
CSV_1HOUR = os.path.join(_BASE_DIR, "Demand_Data_1h.csv")
CSV_WEATHER = os.path.join(_BASE_DIR, "weather_1h.csv")
BATCH_SIZE = 5000

# 컬럼 매핑: CSV → DB
COLUMN_MAP_5MIN = {
    "기준일시": "timestamp",
    "현재수요(MW)": "current_demand",
    "공급능력(MW)": "current_supply",
    "공급예비력(MW)": "supply_reserve",
    "공급예비율(%)": "reserve_rate",
    "운영예비력(MW)": "operation_reserve",
}

COLUMN_MAP_1HOUR = {
    "기준일시": "timestamp",
    "현재수요(MW)_mean": "demand_mean",
    "현재수요(MW)_max": "demand_max",
    "현재수요(MW)_min": "demand_min",
    "공급능력(MW)_mean": "supply_mean",
    "공급능력(MW)_max": "supply_max",
    "공급능력(MW)_min": "supply_min",
    "공급예비력(MW)_mean": "reserve_mean",
    "공급예비력(MW)_max": "reserve_max",
    "공급예비력(MW)_min": "reserve_min",
    "공급예비율(%)_mean": "reserve_rate_mean",
    "공급예비율(%)_max": "reserve_rate_max",
    "공급예비율(%)_min": "reserve_rate_min",
    "운영예비력(MW)_mean": "op_reserve_mean",
    "운영예비력(MW)_max": "op_reserve_max",
    "운영예비력(MW)_min": "op_reserve_min",
    "공휴일": "is_holiday",
    "요일유형": "day_type",
}

# ========================================
# Holiday Utilities
# ========================================
_calendar = SouthKorea()
_holiday_cache: dict[str, bool] = {}


def is_holiday(date: datetime) -> bool:
    date_key = date.strftime("%Y-%m-%d")
    if date_key not in _holiday_cache:
        _holiday_cache[date_key] = _calendar.is_holiday(date.date())
    return _holiday_cache[date_key]


def get_day_type(date: datetime) -> int:
    """0=평일, 1=주말, 2=공휴일"""
    if is_holiday(date):
        return 2
    if date.weekday() >= 5:
        return 1
    return 0


# ========================================
# Load Functions
# ========================================
def load_5min_to_db(csv_path: str = CSV_5MIN) -> int:
    """5분 단위 CSV를 DB에 로드"""
    print(f"\n{'='*60}")
    print(f"5분 데이터 로드: {csv_path} → {DB_PATH}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_path):
        print(f"[ERROR] 파일 없음: {csv_path}")
        return 0

    # DB 초기화
    init_db_sync()

    # CSV 로드
    df = pd.read_csv(csv_path, encoding="euc-kr", on_bad_lines="skip")
    print(f"[INFO] CSV 로드: {len(df):,}건")

    # 컬럼명 변환
    df = df.rename(columns=COLUMN_MAP_5MIN)

    # 필요한 컬럼만 선택
    available_cols = [c for c in COLUMN_MAP_5MIN.values() if c in df.columns]
    df = df[available_cols].copy()

    # timestamp 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 공휴일, 요일유형 계산
    print("[INFO] 공휴일/요일유형 계산 중...")
    df["is_holiday"] = df["timestamp"].apply(is_holiday)
    df["day_type"] = df["timestamp"].apply(get_day_type)

    # NaN 처리
    df = df.where(pd.notnull(df), None)

    # DB에 배치로 저장
    print(f"[INFO] DB 저장 시작 (배치 크기: {BATCH_SIZE})")
    total = len(df)
    saved = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch.to_sql(
            "demand_5min",
            sync_engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        saved += len(batch)
        pct = (saved / total) * 100
        print(f"  [{saved:,}/{total:,}] {pct:.1f}%")

    print(f"\n[DONE] {saved:,}건 저장 완료")
    return saved


def load_1hour_to_db(csv_path: str = CSV_1HOUR) -> int:
    """1시간 집계 CSV를 DB에 로드"""
    print(f"\n{'='*60}")
    print(f"1시간 데이터 로드: {csv_path} → {DB_PATH}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_path):
        print(f"[ERROR] 파일 없음: {csv_path}")
        return 0

    # DB 초기화
    init_db_sync()

    # CSV 로드
    df = pd.read_csv(csv_path, encoding="euc-kr", on_bad_lines="skip")
    print(f"[INFO] CSV 로드: {len(df):,}건")

    # 컬럼명 변환
    df = df.rename(columns=COLUMN_MAP_1HOUR)

    # timestamp 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # NaN 처리
    df = df.where(pd.notnull(df), None)

    # DB에 배치로 저장
    print(f"[INFO] DB 저장 시작 (배치 크기: {BATCH_SIZE})")
    total = len(df)
    saved = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch.to_sql(
            "demand_1hour",
            sync_engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        saved += len(batch)
        pct = (saved / total) * 100
        print(f"  [{saved:,}/{total:,}] {pct:.1f}%")

    print(f"\n[DONE] {saved:,}건 저장 완료")
    return saved


def load_weather_to_db(csv_path: str = CSV_WEATHER) -> int:
    """기상 데이터 CSV를 DB에 로드"""
    print(f"\n{'='*60}")
    print(f"기상 데이터 로드: {csv_path} → {DB_PATH}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_path):
        print(f"[ERROR] 파일 없음: {csv_path}")
        return 0

    # DB 초기화
    init_db_sync()

    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV 로드: {len(df):,}건")

    # timestamp 변환
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # NaN 처리
    df = df.where(pd.notnull(df), None)

    # DB에 배치로 저장
    print(f"[INFO] DB 저장 시작 (배치 크기: {BATCH_SIZE})")
    total = len(df)
    saved = 0

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]
        batch.to_sql(
            "weather_1hour",
            sync_engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        saved += len(batch)
        pct = (saved / total) * 100
        print(f"  [{saved:,}/{total:,}] {pct:.1f}%")

    print(f"\n[DONE] {saved:,}건 저장 완료")
    return saved


def check_db_status():
    """DB 현재 상태 확인"""
    import sqlite3

    print(f"\n{'='*60}")
    print(f"DB 상태 확인: {DB_PATH}")
    print(f"{'='*60}\n")

    if not os.path.exists(DB_PATH):
        print("[INFO] DB 파일 없음")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 5분 데이터
    cur.execute("SELECT COUNT(*) FROM demand_5min")
    cnt_5min = cur.fetchone()[0]
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM demand_5min")
    range_5min = cur.fetchone()

    # 1시간 데이터
    cur.execute("SELECT COUNT(*) FROM demand_1hour")
    cnt_1hour = cur.fetchone()[0]
    cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM demand_1hour")
    range_1hour = cur.fetchone()

    # 기상 데이터
    try:
        cur.execute("SELECT COUNT(*) FROM weather_1hour")
        cnt_weather = cur.fetchone()[0]
        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM weather_1hour")
        range_weather = cur.fetchone()
    except:
        cnt_weather = 0
        range_weather = (None, None)

    conn.close()

    print(f"demand_5min:   {cnt_5min:,}건")
    if range_5min[0]:
        print(f"  기간: {range_5min[0]} ~ {range_5min[1]}")

    print(f"\ndemand_1hour:  {cnt_1hour:,}건")
    if range_1hour[0]:
        print(f"  기간: {range_1hour[0]} ~ {range_1hour[1]}")

    print(f"\nweather_1hour: {cnt_weather:,}건")
    if range_weather[0]:
        print(f"  기간: {range_weather[0]} ~ {range_weather[1]}")


def clear_db():
    """DB 테이블 초기화"""
    import sqlite3

    print("[WARN] DB 테이블 초기화...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM demand_5min")
    cur.execute("DELETE FROM demand_1hour")
    conn.commit()
    conn.close()
    print("[DONE] 초기화 완료")


# ========================================
# CLI
# ========================================
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--status" in args:
        check_db_status()
    elif "--clear" in args:
        clear_db()
    elif "--weather" in args:
        load_weather_to_db()
    elif "--hourly" in args:
        load_1hour_to_db()
    elif "--all" in args:
        load_5min_to_db()
        load_1hour_to_db()
        load_weather_to_db()
    elif "--5min" in args or len(args) == 0:
        load_5min_to_db()
    else:
        print("Usage:")
        print("  python load_csv_to_db.py           # 5분 데이터 로드")
        print("  python load_csv_to_db.py --hourly  # 1시간 데이터 로드")
        print("  python load_csv_to_db.py --all     # 둘 다 로드")
        print("  python load_csv_to_db.py --status  # DB 상태 확인")
        print("  python load_csv_to_db.py --clear   # DB 초기화")
