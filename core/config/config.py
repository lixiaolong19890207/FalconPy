from pathlib import Path

from pydantic import BaseSettings

from common.log import logger


class Settings(BaseSettings):
    """
    服务启动读取配置的class
    """
    PROJECT_NAME: str = "Falcon 3D Engine"
    API_VERSION: str = "/v1"
    BASE_DIR: str = str(Path(__file__).resolve().parent.parent.parent)
    logger.info(f"Project dir is {BASE_DIR}")

    # Token 配置
    SECRET_KEY = '8361D4BEF96ACD2A'  # 加密
    TOKEN_EXPIRATION = 60 * 24  # 有效期: 一天（单位：分钟）

    FIRST_SUPERUSER = 'niuniu'
    FIRST_SUPERUSER_PASSWORD = 'P@ssw0rd'


settings = Settings()
