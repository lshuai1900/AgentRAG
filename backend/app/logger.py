"""日志配置"""
import sys

from loguru import logger

from .config import settings


def setup_logger():
    """配置 loguru 日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        f"{settings.data_dir}/logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
    )
    return logger


log = setup_logger()
