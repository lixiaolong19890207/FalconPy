import os
import sys
from pathlib import Path

from loguru import logger

# Add log storage file  添加日志存储路径
LOG_PATH = Path(__file__).resolve().parent.parent.parent
LOG_PATH = str(LOG_PATH / "logs")


# Log message output format 日志格式配置
file_message = (
    "<g>[{time:YYYY-MM-DD HH:mm:ss:SSS}] [PID:{process.id}]</>"
    " | <lvl>{level}</> | <g>{module}:{line}</> | <lvl>{function}:[{message}]</>"
)


logger.remove()
logger.add(
    sys.stderr,
    format=file_message,  # format about log information
    level="DEBUG",  # log level (Info will include INFO, WARNING and ERROR)
    colorize=True,  # log with the colors
    enqueue=True,  # multiprocess safe
    backtrace=True,
    diagnose=False,
    catch=True,
)

params = {
    "rotation": "20 MB",  # create new log every 20 MB
    "retention": "30 days",  # every 30 days clear log files
    "colorize": False,  # log with the colors
    "encoding": "utf8",
    "compression": "zip",
    "enqueue": True,  # multiprocess safe
    "backtrace": True,
    "diagnose": False,
    "catch": True,
}

logger.add(
    os.path.join(LOG_PATH, "falcon_info.log"),
    format=file_message,  # format about log information
    level="INFO",  # log level (Info will include INFO, WARNING and ERROR)
    **params,
)

logger.add(
    os.path.join(LOG_PATH, "falcon_error.log"),
    format=file_message,  # format about log information
    level="ERROR",  # log level (only ERROR)
    **params,
)


logger: logger
