"""
기상 데이터 시간별 평균 집계

여러 관측소 데이터를 시간별 전국 평균으로 집계
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd

# 경로 설정
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(_BASE_DIR, "asos_all_merged.csv")
OUTPUT_CSV = os.path.join(_BASE_DIR, "weather_1h.csv")


def aggregate_weather_hourly(
    input_path: str = INPUT_CSV,
    output_path: str = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    관측소별 기상 데이터를 시간별 전국 평균으로 집계

    Returns:
        집계된 DataFrame (시간별 평균 기온, 습도)
    """
    print(f"\n{'='*60}")
    print(f"기상 데이터 시간별 집계: {input_path}")
    print(f"{'='*60}\n")

    # 데이터 로드
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일 없음: {input_path}")

    df = pd.read_csv(input_path)
    print(f"[INFO] 원본 데이터: {len(df):,}건")
    print(f"[INFO] 관측소 수: {df['station_name'].nunique()}개")

    # timestamp 파싱
    df["timestamp"] = pd.to_datetime(df["date"])

    # 시간별 전국 평균 집계
    result = df.groupby("timestamp").agg({
        "temperature": ["mean", "max", "min"],
        "humidity": ["mean", "max", "min"],
    })

    # 컬럼명 평탄화
    result.columns = [f"{col}_{agg}" for col, agg in result.columns]
    result = result.reset_index()

    # 소수점 정리
    for col in result.columns:
        if col != "timestamp":
            result[col] = result[col].round(1)

    print(f"[INFO] 집계 결과: {len(result):,}건 (시간별)")
    print(f"[INFO] 기간: {result['timestamp'].min()} ~ {result['timestamp'].max()}")

    # 저장
    result.to_csv(output_path, index=False)
    print(f"[DONE] 저장: {output_path}")

    return result


def check_weather_data():
    """집계된 기상 데이터 확인"""
    if not os.path.exists(OUTPUT_CSV):
        print("[INFO] 집계 파일 없음. 먼저 aggregate_weather_hourly() 실행 필요")
        return

    df = pd.read_csv(OUTPUT_CSV)
    print(f"\n기상 데이터 (시간별 평균)")
    print(f"{'='*50}")
    print(f"총 건수: {len(df):,}")
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"\n최근 5건:")
    print(df.tail())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_weather_data()
    else:
        aggregate_weather_hourly()
