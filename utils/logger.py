"""结构化日志。

所有日志以 JSON 格式输出，方便后续接入 ELK/Loki 等日志系统。
同时支持控制台可读输出（开发模式）和纯 JSON 输出（生产模式）。
"""

import json
import logging
import sys
import time
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """把 log record 格式化为 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 合入 extra 字段（structured data）
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class DevFormatter(logging.Formatter):
    """开发模式的可读格式。"""

    COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = self.formatTime(record, "%H:%M:%S")
        msg = f"{color}{ts} [{record.levelname:7s}] {record.name}: {record.getMessage()}{self.RESET}"
        if hasattr(record, "extra_data"):
            extras = record.extra_data
            if extras:
                # 只展示关键字段，不全部 dump
                compact = " | ".join(f"{k}={v}" for k, v in extras.items())
                msg += f"  ({compact})"
        return msg


def setup_logger(
        name: str = "bci_agent",
        log_dir: str = "logs",
        dev_mode: bool = True,
        level: int = logging.INFO,
) -> logging.Logger:
    """初始化 logger。

    Args:
        name: logger 名
        log_dir: JSON 日志文件目录
        dev_mode: True 时控制台输出可读格式，False 时输出 JSON
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复 handler
    if logger.handlers:
        return logger

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(DevFormatter() if dev_mode else JsonFormatter())
    logger.addHandler(console)

    # 文件 handler（始终 JSON 格式）
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path / f"{name}.jsonl", encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


def log_event(logger: logging.Logger, message: str, level: str = "info", **kwargs):
    """带结构化数据的日志输出。

    用法：
        log_event(logger, "tool call completed",
                  tool_name="rag_search", duration_ms=123, status="ok")
    """
    record = logger.makeRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        fn="", lno=0, msg=message,
        args=(), exc_info=None,
    )
    record.extra_data = kwargs
    logger.handle(record)
