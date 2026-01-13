"""
5분 단위 전력수요 데이터를 1시간 단위로 집계하는 모듈

- 5분 데이터 12개 → 1시간 평균/최대/최소값 계산
- 공휴일, 요일유형 정보 포함
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd
from workalendar.asia import SouthKorea


# ========================================
# Constants
# ========================================
INPUT_CSV = "Demand_Data_all.csv"
OUTPUT_CSV = "Demand_Data_1h.csv"

# 집계 대상 컬럼
DEMAND_COL = "현재수요(MW)"
SUPPLY_COL = "공급능력(MW)"
RESERVE_COL = "공급예비력(MW)"
RESERVE_RATE_COL = "공급예비율(%)"
OP_RESERVE_COL = "운영예비력(MW)"

DATETIME_COL = "기준일시"


# ========================================
# Holiday Check
# ========================================
_calendar = SouthKorea()
_holiday_cache: dict[str, bool] = {}


def is_holiday(date: datetime) -> bool:
    """Check if the given date is a holiday in Korea."""
    date_key = date.strftime("%Y-%m-%d")
    if date_key not in _holiday_cache:
        _holiday_cache[date_key] = _calendar.is_holiday(date.date())
    return _holiday_cache[date_key]


def get_day_type(date: datetime) -> int:
    """
    요일 유형을 반환합니다.

    Returns:
        0: 평일 (월~금, 공휴일 아님)
        1: 주말 (토, 일)
        2: 공휴일 (공휴일 캘린더 기준)
    """
    if is_holiday(date):
        return 2
    if date.weekday() >= 5:
        return 1
    return 0


# ========================================
# Aggregation Functions
# ========================================
def aggregate_5min_to_1h(
    input_path: str = INPUT_CSV,
    output_path: str = OUTPUT_CSV,
    encoding: str = "euc-kr",
    last_timestamp: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    5분 단위 데이터를 1시간 단위로 집계합니다.

    Args:
        input_path: 5분 단위 CSV 파일 경로
        output_path: 1시간 단위 CSV 출력 경로
        encoding: 파일 인코딩
        last_timestamp: 이 시간 이후 데이터만 처리 (백필용)

    Returns:
        집계된 DataFrame
    """
    print(f"\n{'='*60}")
    print(f"5분 → 1시간 집계: {input_path}")
    print(f"{'='*60}\n")

    # 1) 데이터 로드
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    df = pd.read_csv(input_path, encoding=encoding, on_bad_lines="skip")
    print(f"[INFO] 원본 데이터: {len(df):,}건")

    # 2) 타임스탬프 파싱
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

    # 3) 백필: 마지막 timestamp 이후만 처리
    if last_timestamp:
        df = df[df[DATETIME_COL] > last_timestamp]
        print(f"[INFO] 필터 후: {len(df):,}건 (>{last_timestamp})")

    if df.empty:
        print("[INFO] 집계할 데이터 없음")
        return pd.DataFrame()

    # 4) 1시간 단위로 그룹화 (floor to hour)
    df["hour"] = df[DATETIME_COL].dt.floor("h")

    # 5) 집계 대상 컬럼 확인
    agg_cols = [DEMAND_COL, SUPPLY_COL, RESERVE_COL, RESERVE_RATE_COL, OP_RESERVE_COL]
    available_cols = [c for c in agg_cols if c in df.columns]

    if not available_cols:
        raise ValueError(f"집계 대상 컬럼 없음: {agg_cols}")

    print(f"[INFO] 집계 대상 컬럼: {available_cols}")

    # 6) 집계 수행: 평균, 최대, 최소
    agg_dict = {}
    for col in available_cols:
        agg_dict[col] = ["mean", "max", "min"]

    result = df.groupby("hour").agg(agg_dict)

    # 7) 컬럼명 평탄화: (컬럼, 집계함수) -> 컬럼_집계함수
    result.columns = [f"{col}_{agg}" for col, agg in result.columns]
    result = result.reset_index()
    result = result.rename(columns={"hour": DATETIME_COL})

    # 8) 공휴일, 요일유형 추가
    result["공휴일"] = result[DATETIME_COL].apply(lambda x: 1 if is_holiday(x) else 0)
    result["요일유형"] = result[DATETIME_COL].apply(get_day_type)

    print(f"[INFO] 집계 결과: {len(result):,}건 (1시간 단위)")

    # 9) 저장
    # 기존 파일이 있으면 append, 없으면 새로 생성
    if last_timestamp and os.path.exists(output_path):
        # 백필 모드: append
        result.to_csv(
            output_path,
            mode="a",
            header=False,
            index=False,
            encoding=encoding,
        )
        print(f"[DONE] 추가 저장: {output_path}")
    else:
        result.to_csv(output_path, index=False, encoding=encoding)
        print(f"[DONE] 저장: {output_path}")

    return result


def get_last_timestamp_from_output(
    output_path: str = OUTPUT_CSV,
    encoding: str = "euc-kr",
) -> Optional[datetime]:
    """
    출력 파일에서 마지막 timestamp를 읽습니다.
    """
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return None

    try:
        df = pd.read_csv(output_path, encoding=encoding, usecols=[DATETIME_COL])
        if df.empty:
            return None
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
        return df[DATETIME_COL].max()
    except Exception as e:
        print(f"[WARN] 마지막 timestamp 읽기 실패: {e}")
        return None


def run_with_backfill(
    input_path: str = INPUT_CSV,
    output_path: str = OUTPUT_CSV,
    encoding: str = "euc-kr",
) -> pd.DataFrame:
    """
    백필 모드로 집계를 수행합니다.
    출력 파일의 마지막 timestamp 이후 데이터만 처리합니다.
    """
    last_ts = get_last_timestamp_from_output(output_path, encoding)
    if last_ts:
        print(f"[INFO] 마지막 집계 시간: {last_ts}")
    else:
        print("[INFO] 출력 파일 없음. 전체 집계 수행")

    return aggregate_5min_to_1h(
        input_path=input_path,
        output_path=output_path,
        encoding=encoding,
        last_timestamp=last_ts,
    )


# ========================================
# CLI Entry Point
# ========================================
if __name__ == "__main__":
    import sys

    print("전력수요 1시간 집계기")
    print("=" * 40)
    print("1. 전체 집계 (처음부터)")
    print("2. 백필 모드 (마지막 이후만)")
    print("=" * 40)

    choice = input("선택 (1-2): ").strip()

    if choice == "1":
        # 전체 집계
        aggregate_5min_to_1h()
    elif choice == "2":
        # 백필 모드
        run_with_backfill()
    else:
        print("잘못된 선택")
        sys.exit(1)
