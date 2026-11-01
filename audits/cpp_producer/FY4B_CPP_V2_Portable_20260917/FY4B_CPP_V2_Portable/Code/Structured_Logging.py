"""多进程安全的结构化日志基础设施。

设计原则：

* 控制台保留简洁的人类可读日志；
* 每个进程写独立 JSONL 文件，避免多个进程争抢同一 FileHandler；
* 所有记录包含 run_id、event、时间、进程和业务字段；
* 子进程从环境变量继承同一个 run_id 和日志目录。

一次流水线的日志位于 ``<log_dir>/<run_id>/events.<pid>.jsonl``，可直接由
日志采集系统按 JSON Lines 读取，也可用 jq/PowerShell/Python 聚合分析。

版本历史：
    V1.1_122+local.1  新增每进程 JSONL、控制台日志和 run_id 跨进程传播，并补充
                         字段、重复初始化及序列化降级策略注释。
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_STANDARD_RECORD_FIELDS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


def _json_safe(value: Any) -> Any:
    """把 datetime、Path、numpy 标量等不可直接序列化对象转成稳定文本。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


class JsonLineFormatter(logging.Formatter):
    """将 ``LogRecord`` 转换成一行 UTF-8 JSON。

    固定字段用于跨任务检索，调用方通过 ``extra`` 传入的业务字段会保留；异常
    堆栈写入 ``exception``。每条记录严格占一行，方便采集器断点续读。
    """

    def format(self, record: logging.LogRecord) -> str:
        """构造固定字段、补充业务字段并返回紧凑 JSON 文本。"""
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log_message"),
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", os.getenv("GLOBAL_CPP_RUN_ID", "standalone")),
            "host": socket.gethostname(),
            "pid": record.process,
            "process_name": record.processName,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and key not in payload:
                payload[key] = _json_safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """控制台格式：保留 event，便于操作人员快速定位阶段。"""

    def format(self, record: logging.LogRecord) -> str:
        """返回面向操作人员的本地时间、级别、事件名和消息。"""
        event_name = getattr(record, "event", "log_message")
        stamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"[{stamp}] {record.levelname:<7} {event_name}: {record.getMessage()}"


def make_run_id(prefix: str = "global_cpp") -> str:
    """生成可排序、可跨进程传播的运行标识。"""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    return f"{prefix}_{stamp}_{os.getpid()}"


def configure_structured_logging(
    log_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    level: Optional[str] = None,
    console: bool = True,
) -> Path:
    """初始化当前进程日志并返回本进程 JSONL 路径。

    每个进程只配置一次。重复调用不会叠加 handler，适合统一入口和 worker
    函数都调用。日志目录和 run_id 会写入环境变量供 spawn 子进程继承。

    ``level`` 接受 logging 级别名称，未知名称降级为 INFO。函数只移除自己创建
    且带标记的 handler，不干扰宿主程序的其它日志配置。返回路径格式为
    ``<log_dir>/<run_id>/events.<pid>.jsonl``。
    """
    resolved_run_id = run_id or os.getenv("GLOBAL_CPP_RUN_ID") or make_run_id()
    resolved_log_dir = Path(
        log_dir or os.getenv("GLOBAL_CPP_LOG_DIR") or (Path(__file__).resolve().parent.parent / "Logs")
    ).expanduser().resolve()
    resolved_level = (level or os.getenv("GLOBAL_CPP_LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)

    os.environ["GLOBAL_CPP_RUN_ID"] = resolved_run_id
    os.environ["GLOBAL_CPP_LOG_DIR"] = str(resolved_log_dir)
    os.environ["GLOBAL_CPP_LOG_LEVEL"] = resolved_level

    run_dir = resolved_log_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"events.{os.getpid()}.jsonl"

    root = logging.getLogger()
    configuration_key = (str(json_path), resolved_level, console)
    if getattr(root, "_global_cpp_configuration", None) == configuration_key:
        return json_path

    for handler in list(root.handlers):
        if getattr(handler, "_global_cpp_handler", False):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(numeric_level)
    file_handler = logging.FileHandler(json_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(JsonLineFormatter())
    file_handler._global_cpp_handler = True
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(HumanFormatter())
        console_handler._global_cpp_handler = True
        root.addHandler(console_handler)

    root._global_cpp_configuration = configuration_key
    return json_path


def get_logger(name: str) -> logging.Logger:
    """返回业务 logger；若尚未初始化则使用 logging 默认行为。"""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event_name: str,
    message: str,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """写一条带业务字段的结构化事件。

    ``fields`` 中的 datetime、Path、容器和 numpy 标量会先转换成 JSON 安全值；
    ``exc_info=True`` 时由 formatter 附加当前异常堆栈。若业务字段与 logging 的
    ``module/name/message`` 等保留属性重名，会自动加 ``field_`` 前缀，避免
    ``LogRecord`` 因属性覆盖抛出 ``KeyError`` 并中断正常业务流程。
    """
    safe_fields = {}
    for key, value in fields.items():
        output_key = f"field_{key}" if key in _STANDARD_RECORD_FIELDS else key
        safe_fields[output_key] = _json_safe(value)
    extra = {
        "event": event_name,
        "run_id": os.getenv("GLOBAL_CPP_RUN_ID", "standalone"),
        **safe_fields,
    }
    logger.log(level, message, extra=extra, exc_info=exc_info)
