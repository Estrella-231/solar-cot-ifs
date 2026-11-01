"""
Description: 根据时次选择卫星并并行调度各卫星反演任务。
Author: Yangzx
LastEditTime: 2026-07-17
Version:
    V1.1_122+local.1
Change:
    1. V1.0: 提供按时次选择卫星并并行调度反演任务的功能。
    2. V1.1: 修复 BrokenProcessPool 导致全部后续时次级联失败的问题；
             新增对已损坏进程池的检测与自动重建；
             重构 Run_Retrievals 加入双重容错机制。
    3. local.1: 卫星更替规则迁移到 Satellite_Timeline 单一配置源；
                新增 Run_Selected_Retrievals，支持任意单星/多星组合；
                卫星适配器改为按需导入，单星任务不再加载全部卫星资源。
    4. local.1: 调度、进程池重建、卫星开始/完成/失败统一记录结构化事件。
    5. local.1: 补充按需导入、进程池生命周期和失败隔离说明；调度逻辑不变。
"""

import multiprocessing as mp
import importlib
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from Satellite_Timeline import get_active_satellites, validate_satellites
from Structured_Logging import configure_structured_logging, get_logger, log_event

LOGGER = get_logger("global_cpp.scheduler")


Retrieval_Functions: Dict[str, Tuple[str, str]] = {
    "FY4A_105E": ("Retrieval_FY4A_105E", "Retrieval_FY4A_105E_CPP"),
    "FY4B_105E": ("Retrieval_FY4B_105E", "Retrieval_FY4B_105E_CPP"),
    "FY4B_133E": ("Retrieval_FY4B_133E", "Retrieval_FY4B_133E_CPP"),
    "GOES16": ("Retrieval_GOES16", "Retrieval_GOES16_CPP"),
    "GOES17": ("Retrieval_GOES17", "Retrieval_GOES17_CPP"),
    "GOES18": ("Retrieval_GOES18", "Retrieval_GOES18_CPP"),
    "MSG1_IODC": ("Retrieval_MSG1_IODC", "Retrieval_MSG1_IODC_CPP"),
    "MSG2_IODC": ("Retrieval_MSG2_IODC", "Retrieval_MSG2_IODC_CPP"),
    "MSG3_0DEG": ("Retrieval_MSG3_0DEG", "Retrieval_MSG3_0DEG_CPP"),
    "MSG4_0DEG": ("Retrieval_MSG4_0DEG", "Retrieval_MSG4_0DEG_CPP"),
}
_Retrieval_Function_Cache: Dict[str, Callable] = {}

# 全局进程池（可复用，需配合 BrokenProcessPool 检测使用）
Retrieval_Executor: Optional[ProcessPoolExecutor] = None
Retrieval_Executor_Workers: Optional[int] = None
_SPAWN_CTX = mp.get_context("spawn")

def Call_Retrieval_Function(Satellite_Name: str, Run_Time: datetime) -> None:
    """按需导入并调用单颗卫星函数，避免单星任务加载全部卫星资源。"""
    Function = _Retrieval_Function_Cache.get(Satellite_Name)
    if Function is None:
        try:
            Module_Name, Function_Name = Retrieval_Functions[Satellite_Name]
        except KeyError as Exc:
            # 只把卫星映射查询失败解释为未知卫星。导入和反演内部的 KeyError
            # 必须保留原异常，避免把真实配置/数据问题误报为卫星名称错误。
            raise ValueError(f"未知卫星: {Satellite_Name}") from Exc
        Function = getattr(importlib.import_module(Module_Name), Function_Name)
        _Retrieval_Function_Cache[Satellite_Name] = Function
        log_event(
            LOGGER, "satellite_adapter_loaded", "卫星适配器按需加载完成",
            satellite=Satellite_Name, adapter_module=Module_Name,
            adapter_function=Function_Name,
        )
    Function(Run_Time)


def Get_Retrieval_Satellites(Run_Time: datetime) -> List[str]:
    """根据共享时间线选择生产全球产品所需的卫星列表。"""
    return list(get_active_satellites(Run_Time))


def Run_One_Retrieval(Satellite_Name: str, Run_Time: datetime) -> Dict:
    """执行单颗卫星任务并返回统一结果结构。"""
    configure_structured_logging(console=True)
    started = time.perf_counter()
    try:
        log_event(
            LOGGER, "satellite_retrieval_started", "开始卫星反演",
            satellite=Satellite_Name, run_time=Run_Time,
        )
        Call_Retrieval_Function(Satellite_Name, Run_Time)
        elapsed = time.perf_counter() - started
        log_event(
            LOGGER, "satellite_retrieval_finished", "卫星反演完成",
            satellite=Satellite_Name, run_time=Run_Time,
            status="SUCCESS", elapsed_seconds=round(elapsed, 3),
        )
        return {
            "Run_Time": Run_Time,
            "Satellite_Name": Satellite_Name,
            "Status": "SUCCESS",
            "Error": "",
        }
    except Exception as Exc:
        Error_Info = traceback.format_exc()
        log_event(
            LOGGER, "satellite_retrieval_failed", "卫星反演失败",
            level=logging.ERROR, exc_info=True,
            satellite=Satellite_Name, run_time=Run_Time,
            status="ERROR", elapsed_seconds=round(time.perf_counter() - started, 3),
            error=repr(Exc),
        )
        return {
            "Run_Time": Run_Time,
            "Satellite_Name": Satellite_Name,
            "Status": "ERROR",
            "Error": Error_Info,
        }


def _Is_Executor_Broken(Executor: Optional[ProcessPoolExecutor]) -> bool:
    """
    检测进程池是否已损坏。
    ProcessPoolExecutor 在子进程异常终止时设置 _broken 标志，
    之后任何 submit() 都会抛出 BrokenProcessPool。
    """
    if Executor is None:
        return True
    # 通过属性直接检测内部 _broken 标志（Python 3.9+ 实现细节）
    # 安全做法：先尝试 submit 一个空任务来验证
    return getattr(Executor, '_broken', False)


def Initialize_Retrieval_Pool(Max_Workers: int) -> ProcessPoolExecutor:
    """
    初始化/重建可复用的卫星反演进程池。
    【V1.1】新增对已损坏进程池的检测，自动重建。
    """
    global Retrieval_Executor, Retrieval_Executor_Workers

    if Max_Workers <= 0:
        raise ValueError("Max_Workers 必须大于 0")

    # 如果现有池子可用且参数匹配，直接复用
    if (Retrieval_Executor is not None
            and Retrieval_Executor_Workers == Max_Workers
            and not _Is_Executor_Broken(Retrieval_Executor)):
        return Retrieval_Executor

    # 池子不存在 / 参数变化 / 已损坏 → 重建
    Shutdown_Retrieval_Pool()
    Retrieval_Executor = ProcessPoolExecutor(
        max_workers=Max_Workers, mp_context=_SPAWN_CTX
    )
    Retrieval_Executor_Workers = Max_Workers
    log_event(
        LOGGER, "retrieval_pool_initialized", "卫星反演进程池已初始化",
        max_workers=Max_Workers,
    )
    return Retrieval_Executor


def Shutdown_Retrieval_Pool():
    """关闭当前可复用的卫星反演进程池。"""
    global Retrieval_Executor, Retrieval_Executor_Workers
    if Retrieval_Executor is not None:
        Retrieval_Executor.shutdown(wait=True)
        Retrieval_Executor = None
        Retrieval_Executor_Workers = None
        log_event(LOGGER, "retrieval_pool_shutdown", "卫星反演进程池已关闭")


def Run_Selected_Retrievals(
    Run_Time: datetime,
    Satellite_Names: Sequence[str],
    Max_Workers: Optional[int] = None,
) -> List[Dict]:
    """
    并行执行某一时次下显式指定的单星或多星任务。

    local.1 新增：保留 V1.1 的进程池损坏检测、自动重建和重试机制，
    同时允许统一 CLI 调度单颗卫星、任意多颗卫星或自动生产组合。
    """
    Satellite_List = list(validate_satellites(Satellite_Names))
    configure_structured_logging(console=True)
    log_event(
        LOGGER, "retrieval_group_started", "开始当前时次卫星组反演",
        run_time=Run_Time, satellites=Satellite_List,
    )

    if Max_Workers is None:
        Max_Workers = len(Satellite_List)
    Max_Workers = min(Max_Workers, len(Satellite_List))

    # ---- 最多重试 2 次（第 1 次失败后重建池子再试 1 次） ----
    Max_Retries = 2
    Last_Error = None

    for Attempt in range(1, Max_Retries + 1):
        try:
            Executor = Initialize_Retrieval_Pool(Max_Workers)
            Future_Map = {
                Executor.submit(Run_One_Retrieval, Satellite_Name, Run_Time): Satellite_Name
                for Satellite_Name in Satellite_List
            }

            Result_List = []
            for Future in as_completed(Future_Map):
                Result_List.append(Future.result())

            # 成功：按原顺序排序后返回
            Name_Order = {Name: Index for Index, Name in enumerate(Satellite_List)}
            Result_List.sort(key=lambda X: Name_Order[X["Satellite_Name"]])
            failed = [item["Satellite_Name"] for item in Result_List if item["Status"] != "SUCCESS"]
            log_event(
                LOGGER, "retrieval_group_finished", "当前时次卫星组反演结束",
                level=logging.INFO if not failed else logging.ERROR,
                run_time=Run_Time, satellites=Satellite_List,
                status="SUCCESS" if not failed else "ERROR", failed_satellites=failed,
            )
            return Result_List

        except BrokenProcessPool as Exc:
            Last_Error = Exc
            log_event(
                LOGGER, "retrieval_pool_broken", "卫星反演进程池损坏",
                level=logging.ERROR, run_time=Run_Time,
                attempt=Attempt, max_retries=Max_Retries, error=repr(Exc),
            )
            Shutdown_Retrieval_Pool()  # 销毁损坏池子
            if Attempt < Max_Retries:
                log_event(
                    LOGGER, "retrieval_pool_retry", "重建进程池并重试卫星组",
                    level=logging.WARNING, run_time=Run_Time, next_attempt=Attempt + 1,
                )
                continue
            # 最后一次尝试也失败了
            raise RuntimeError(
                f"Run_Retrievals 在 {Max_Retries} 次尝试后仍然失败: {Last_Error}"
            ) from Last_Error

        except Exception:
            # 非 BrokenProcessPool 的其他异常不重试，直接抛出
            raise

    # 理论上不会执行到这里
    raise RuntimeError(f"Run_Retrievals 异常退出 (卫星: {Satellite_List})")


def Run_Retrievals(Run_Time: datetime, Max_Workers: Optional[int] = None) -> List[Dict]:
    """按共享卫星时间线反演当前全球产品所需的全部卫星。"""
    return Run_Selected_Retrievals(
        Run_Time,
        Get_Retrieval_Satellites(Run_Time),
        Max_Workers=Max_Workers,
    )
