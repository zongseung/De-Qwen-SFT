"""
설정 관리
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # 프로젝트 경로
    project_root: Path = Path(__file__).parent.parent.parent

    # 데이터 경로
    data_dir: Path = Path(__file__).parent.parent / "data"
    md_file_dir: Path = Path(__file__).parent.parent.parent / "md_file"

    # MCP 서버 설정
    mcp_host: str = "localhost"
    mcp_port: int = 8001

    # LLM 서버 설정
    llm_server_url: str = "http://localhost:8000"
    llm_model_path: str = "./power_demand_merged_model"

    # 외부 API 설정 (기상청 등)
    weather_api_url: str = ""
    weather_api_key: str = ""

    # DB 설정 (SQLite 또는 PostgreSQL)
    database_url: str = "sqlite:///./demand.db"

    # LangGraph 설정
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()
