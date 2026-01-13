"""
전력수요 데이터용 로컬 SQLite 데이터베이스 모듈

- SQLite 파일 기반 (로컬에서 바로 확인 가능)
- 비동기 지원 (aiosqlite)
- 5분 단위 및 1시간 단위 테이블
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ========================================
# Database Configuration
# ========================================
# 로컬 SQLite 파일 경로
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "demand.db")

# SQLite URLs
SYNC_DATABASE_URL = f"sqlite:///{DB_PATH}"
ASYNC_DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# SQLAlchemy Base
Base = declarative_base()


# ========================================
# Models
# ========================================
class Demand5Min(Base):
    """5분 단위 전력수요 데이터"""
    __tablename__ = "demand_5min"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False, index=True)
    current_demand = Column(Float, nullable=True)      # 현재수요(MW)
    current_supply = Column(Float, nullable=True)      # 공급능력(MW)
    supply_capacity = Column(Float, nullable=True)     # 최대예측수요(MW)
    supply_reserve = Column(Float, nullable=True)      # 공급예비력(MW)
    reserve_rate = Column(Float, nullable=True)        # 공급예비율(%)
    operation_reserve = Column(Float, nullable=True)   # 운영예비력(MW)
    is_holiday = Column(Boolean, default=False)        # 공휴일 여부
    day_type = Column(Integer, default=0)              # 0=평일, 1=주말, 2=공휴일

    def __repr__(self):
        return f"<Demand5Min({self.timestamp}, demand={self.current_demand})>"


class Weather1Hour(Base):
    """1시간 단위 기상 데이터 (전국 평균)"""
    __tablename__ = "weather_1hour"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False, index=True)

    temperature_mean = Column(Float, nullable=True)
    temperature_max = Column(Float, nullable=True)
    temperature_min = Column(Float, nullable=True)
    humidity_mean = Column(Float, nullable=True)
    humidity_max = Column(Float, nullable=True)
    humidity_min = Column(Float, nullable=True)

    def __repr__(self):
        return f"<Weather1Hour({self.timestamp}, temp={self.temperature_mean})>"


class Demand1Hour(Base):
    """1시간 단위 전력수요 데이터 (집계)"""
    __tablename__ = "demand_1hour"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, unique=True, nullable=False, index=True)

    # 수요 집계
    demand_mean = Column(Float, nullable=True)
    demand_max = Column(Float, nullable=True)
    demand_min = Column(Float, nullable=True)

    # 공급능력 집계
    supply_mean = Column(Float, nullable=True)
    supply_max = Column(Float, nullable=True)
    supply_min = Column(Float, nullable=True)

    # 예비력 집계
    reserve_mean = Column(Float, nullable=True)
    reserve_max = Column(Float, nullable=True)
    reserve_min = Column(Float, nullable=True)

    # 예비율 집계
    reserve_rate_mean = Column(Float, nullable=True)
    reserve_rate_max = Column(Float, nullable=True)
    reserve_rate_min = Column(Float, nullable=True)

    # 운영예비력 집계
    op_reserve_mean = Column(Float, nullable=True)
    op_reserve_max = Column(Float, nullable=True)
    op_reserve_min = Column(Float, nullable=True)

    is_holiday = Column(Boolean, default=False)
    day_type = Column(Integer, default=0)

    def __repr__(self):
        return f"<Demand1Hour({self.timestamp}, demand_mean={self.demand_mean})>"


# ========================================
# Engine & Session
# ========================================
# Sync engine (for init)
sync_engine = create_engine(SYNC_DATABASE_URL, echo=False)

# Async engine
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

# Async session factory
AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ========================================
# Database Functions
# ========================================
def init_db_sync():
    """동기 방식으로 DB 테이블 생성"""
    Base.metadata.create_all(sync_engine)
    print(f"[DB] 테이블 생성 완료: {DB_PATH}")


async def init_db():
    """비동기 방식으로 DB 테이블 생성"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"[DB] 테이블 생성 완료: {DB_PATH}")


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """비동기 세션 컨텍스트 매니저"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db_path() -> str:
    """DB 파일 경로 반환"""
    return DB_PATH


def db_exists() -> bool:
    """DB 파일 존재 여부"""
    return os.path.exists(DB_PATH)


# ========================================
# CLI for testing
# ========================================
if __name__ == "__main__":
    import asyncio

    print("전력수요 DB 초기화")
    print(f"DB 경로: {DB_PATH}")

    # 동기 방식으로 테이블 생성
    init_db_sync()

    # 테이블 확인
    from sqlalchemy import inspect
    inspector = inspect(sync_engine)
    tables = inspector.get_table_names()
    print(f"생성된 테이블: {tables}")
