#!/usr/bin/env python3
"""Global CPP 统一命令行执行入口。

执行能力：
    * 接受一个或多个 ``--time``，或 ``--start/--end/--freq`` 时间范围；
    * 单星模式可显式选择一颗/多颗卫星，也可用 ``auto`` 按时间线选择；
    * 全球模式可“先完成当前时次全部单星反演再拼接”，或只拼接已有产品；
    * ``--time-workers`` 控制时次并发，单时次内部的卫星并发由相应参数控制；
    * ``plan`` 只解析计划，不读取数据，便于核对卫星随时间更替的组合。

执行边界：
    时间统一转换为无时区 UTC 语义。默认拒绝执行不属于该时次生产组合的卫星，
    历史补算需显式传入 ``--allow-inactive``。每个子进程写独立 JSONL 日志文件，
    最终摘要采用临时文件替换，避免中断时留下半写 JSON。

版本历史：
    V1.1_122+local.1  新增统一单/多时刻、单/多卫星与全球产品入口；增加预检、
                         结构化日志、并发执行、原子摘要及详细函数契约注释。
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Satellite_Timeline import (
    get_active_satellites,
    get_period,
    inactive_satellites,
    validate_satellites,
)
from Version import LOCAL_VERSION
from Structured_Logging import (
    configure_structured_logging,
    get_logger,
    log_event,
    make_run_id,
)

LOGGER = get_logger("global_cpp.pipeline")


def parse_datetime(value: str) -> datetime:
    """解析 ISO 风格时间并统一为无时区 datetime。"""
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"无法解析时间 {value!r}: {exc}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.to_pydatetime()


def build_time_list(args) -> List[datetime]:
    """从可重复 --time 或 --start/--end/--freq 构造时次列表。"""
    if args.time:
        if args.start or args.end:
            raise ValueError("--time 不能与 --start/--end 同时使用")
        return sorted(set(args.time))
    if not args.start or not args.end:
        raise ValueError("请提供 --time，或同时提供 --start 和 --end")
    if args.end < args.start:
        raise ValueError("--end 不能早于 --start")
    try:
        values = pd.date_range(args.start, args.end, freq=args.freq)
    except Exception as exc:
        raise ValueError(f"无效时间频率 {args.freq!r}: {exc}") from exc
    return [value.to_pydatetime() for value in values]


def selected_for_time(
    run_time: datetime,
    requested: Sequence[str],
    allow_inactive: bool,
) -> Tuple[str, ...]:
    """将 auto 或显式名称解析成当前时次的实际卫星列表。"""
    upper = tuple(name.upper() for name in requested)
    if "AUTO" in upper:
        if len(upper) != 1:
            raise ValueError("auto 不能与显式卫星名称混用")
        return get_active_satellites(run_time)
    selected = validate_satellites(upper)
    inactive = inactive_satellites(run_time, selected)
    if inactive and not allow_inactive:
        raise ValueError(
            f"{run_time:%Y-%m-%d %H:%M} 非生产组合卫星: {', '.join(inactive)}；"
            "如确需补算，请增加 --allow-inactive"
        )
    return selected


def make_plan(times: Sequence[datetime], product: str, requested: Sequence[str], allow_inactive: bool) -> List[Dict]:
    """把用户参数展开为逐时次、可序列化的执行计划。

    全球产品始终使用时间线规定的完整卫星组合；单星产品才使用 ``requested``。
    返回字典中的时间采用 ISO 字符串，既可直接打印，也可安全传给工作进程。
    """
    plan = []
    for run_time in times:
        satellites = (get_active_satellites(run_time) if product == "global"
                      else selected_for_time(run_time, requested, allow_inactive))
        period = get_period(run_time)
        plan.append({
            "run_time": run_time.isoformat(),
            "period": period.name,
            "product": product,
            "satellites": list(satellites),
        })
    return plan


def print_plan(plan: Sequence[Dict]) -> None:
    """以稳定的单行格式输出计划，便于人工检查或 shell 管道处理。"""
    for item in plan:
        print(
            f"{item['run_time']} | {item['period']} | {item['product']} | "
            f"{','.join(item['satellites'])}"
        )


def _run_satellite_payload(payload: Dict) -> Dict:
    """进程工作函数：执行一个时次的单星/多星任务。"""
    configure_structured_logging(
        payload["log_dir"], payload["run_id"], payload["log_level"], console=True
    )
    os.environ["GLOBAL_CPP_RETRIEVAL_POLICY"] = payload["retrieval_policy"]
    from Satellite_Choose import Run_Selected_Retrievals, Shutdown_Retrieval_Pool

    run_time = datetime.fromisoformat(payload["run_time"])
    log_event(
        LOGGER, "satellite_time_started", "开始单星/多星时次任务",
        run_time=run_time, satellites=payload["satellites"],
    )
    try:
        results = Run_Selected_Retrievals(
            run_time,
            payload["satellites"],
            Max_Workers=payload["satellite_workers"],
        )
        status = "SUCCESS" if all(item["Status"] == "SUCCESS" for item in results) else "ERROR"
        result = {
            "run_time": payload["run_time"],
            "product": "satellite",
            "status": status,
            "results": [{**item, "Run_Time": str(item["Run_Time"])} for item in results],
        }
        log_event(
            LOGGER, "satellite_time_finished", "单星/多星时次任务结束",
            level=logging.INFO if status == "SUCCESS" else logging.ERROR,
            run_time=run_time, status=status,
        )
        return result
    except Exception as exc:
        log_event(
            LOGGER, "satellite_time_failed", "单星/多星时次任务异常",
            level=logging.ERROR, exc_info=True, run_time=run_time, error=repr(exc),
        )
        return {"run_time": payload["run_time"], "product": "satellite", "status": "ERROR", "error": repr(exc)}
    finally:
        Shutdown_Retrieval_Pool()


def _run_global_payload(payload: Dict) -> Dict:
    """进程工作函数：执行一个时次的全球拼接产品。"""
    configure_structured_logging(
        payload["log_dir"], payload["run_id"], payload["log_level"], console=True
    )
    os.environ["GLOBAL_CPP_RETRIEVAL_POLICY"] = payload["retrieval_policy"]
    os.environ["GLOBAL_CPP_COMPOSITE_POLICY"] = payload["composite_policy"]
    from Test_CON import (
        Border_N, Border_S, NOP, Save_Dir, Global_XLON, Global_YLAT, Subset_Config,
        Process_One_Time, _Resolve_GOES_W_Dir, _Resolve_GOES_E_Dir,
        _Resolve_MSG0_Dir, _Resolve_MSGI_Dir, Shutdown_Retrieval_Pool, _Force_HDF5_Cleanup,
    )

    run_time = datetime.fromisoformat(payload["run_time"])
    log_event(
        LOGGER, "global_time_started", "开始全球产品时次任务",
        run_time=run_time, skip_retrieval=payload["skip_retrieval"],
        satellites=payload["satellites"],
    )
    try:
        time_str, status = Process_One_Time(
            run_time,
            _Resolve_GOES_W_Dir(run_time), _Resolve_GOES_E_Dir(run_time),
            _Resolve_MSG0_Dir(run_time), _Resolve_MSGI_Dir(run_time),
            Border_N, Border_S, NOP,
            Save_Dir, Global_XLON, Global_YLAT, Subset_Config,
            Run_Retrieval=not payload["skip_retrieval"],
            Require_All_Retrievals=True,
            Retrieval_Max_Workers=payload["satellite_workers"],
            Composite_Policy=payload["composite_policy"],
        )
        result = {"run_time": payload["run_time"], "product": "global", "status": status, "output_time": time_str}
        log_event(
            LOGGER, "global_time_finished", "全球产品时次任务结束",
            level=logging.INFO if status == "SUCCESS" else logging.ERROR,
            run_time=run_time, status=status,
        )
        return result
    except Exception as exc:
        log_event(
            LOGGER, "global_time_failed", "全球产品时次任务异常",
            level=logging.ERROR, exc_info=True, run_time=run_time, error=repr(exc),
        )
        return {"run_time": payload["run_time"], "product": "global", "status": "ERROR", "error": repr(exc)}
    finally:
        Shutdown_Retrieval_Pool()
        _Force_HDF5_Cleanup()


def execute_payloads(payloads: Sequence[Dict], worker, time_workers: int) -> List[Dict]:
    """按时间维度串行或并行执行，并保持结果按输入时次排序。"""
    if time_workers <= 0:
        raise ValueError("--time-workers 必须大于 0")
    if time_workers == 1:
        return [worker(payload) for payload in payloads]
    context = mp.get_context("spawn")
    indexed = {}
    with ProcessPoolExecutor(max_workers=time_workers, mp_context=context) as executor:
        for index, payload in enumerate(payloads):
            indexed[executor.submit(worker, payload)] = index
        results = [None] * len(payloads)
        for future in as_completed(indexed):
            results[indexed[future]] = future.result()
    return results


def write_summary(results: Sequence[Dict], output: str = None) -> Path:
    """原子保存机器可读执行摘要；默认写入本地 Logs。"""
    if output:
        path = Path(output).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = PROJECT_ROOT / "Logs" / f"pipeline_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(list(results), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)
    return path


def add_time_arguments(parser: argparse.ArgumentParser) -> None:
    """向子命令注册互斥语义的单时刻与时间范围参数。"""
    parser.add_argument("--time", action="append", type=parse_datetime, help="单时刻；可重复指定")
    parser.add_argument("--start", type=parse_datetime, help="时间范围起点（包含）")
    parser.add_argument("--end", type=parse_datetime, help="时间范围终点（包含）")
    parser.add_argument("--freq", default="60min", help="时间步长，默认 60min")


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """注册并发、日志、结果摘要和运行前检查等公共执行参数。"""
    parser.add_argument("--time-workers", type=int, default=1, help="同时处理的时次数；默认 1")
    parser.add_argument("--satellite-workers", type=int, default=5, help="每时次同时反演的卫星数；默认 5")
    parser.add_argument("--device", help="如 cpu、cuda:0；设置 GLOBAL_CPP_DEVICE")
    parser.add_argument("--no-preflight", action="store_true", help="跳过依赖/路径/模型预检查")
    parser.add_argument("--summary", help="执行摘要 JSON 路径")
    parser.add_argument("--log-dir", help="结构化日志根目录，默认 V2/Logs")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--run-id", help="自定义本次运行标识；默认自动生成")
    parser.add_argument(
        "--retrieval-policy", choices=("skip", "overwrite", "repair"),
        default="skip", help="单星反演输出策略，默认 skip",
    )
    parser.add_argument(
        "--composite-policy", choices=("skip", "overwrite", "repair"),
        default="skip", help="全球拼接输出策略，默认 skip",
    )


def build_parser() -> argparse.ArgumentParser:
    """构建 ``plan``、``satellite``、``global`` 三类子命令解析器。"""
    parser = argparse.ArgumentParser(description=f"Global CPP 统一流水线 {LOCAL_VERSION}")
    parser.add_argument("--version", action="version", version=LOCAL_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="仅显示卫星更替后的执行计划")
    add_time_arguments(plan)
    plan.add_argument("--product", choices=("satellite", "global"), default="global")
    plan.add_argument("--satellite", nargs="+", default=["auto"])
    plan.add_argument("--allow-inactive", action="store_true")
    plan.add_argument("--json", action="store_true", help="以 JSON 输出")

    sat = sub.add_parser("satellite", help="反演单颗或多颗卫星产品")
    add_time_arguments(sat); add_execution_arguments(sat)
    sat.add_argument("--satellite", nargs="+", default=["auto"], help="卫星名列表或 auto")
    sat.add_argument("--allow-inactive", action="store_true", help="允许补算非当前生产组合卫星")

    glob = sub.add_parser("global", help="生成全球拼接产品")
    add_time_arguments(glob); add_execution_arguments(glob)
    glob.add_argument("--skip-retrieval", action="store_true", help="直接拼接已有单星结果")

    check = sub.add_parser("check", help="只读检查单星反演或全球拼接完整性")
    add_time_arguments(check)
    check.add_argument("kind_pos", nargs="?", choices=("retrieval", "composite", "all"))
    check.add_argument("--kind", dest="kind_flag", choices=("retrieval", "composite", "all"), default=None)
    check.add_argument("--report-dir", help="检查报告目录，默认 V2/Test/Check")
    return parser


def main(argv: Sequence[str] = None) -> int:
    """解析参数并执行计划，成功返回 0，参数/预检/任务失败返回非零状态。"""
    args = build_parser().parse_args(argv)
    if args.command == "check":
        try:
            times = build_time_list(args)
        except ValueError as exc:
            print(f"参数错误: {exc}", file=sys.stderr)
            return 2
        from Check.integrity import run_check
        kind = args.kind_pos or args.kind_flag or "all"
        kinds = ("retrieval", "composite") if kind == "all" else (kind,)
        statuses = [run_check(kind, times, args.report_dir) for kind in kinds]
        return 0 if all(status == 0 for status in statuses) else 1
    try:
        times = build_time_list(args)
        requested = getattr(args, "satellite", ["auto"])
        allow_inactive = getattr(args, "allow_inactive", False)
        product = getattr(args, "product", args.command)
        plan = make_plan(times, product, requested, allow_inactive)
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    if args.command == "plan":
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            print_plan(plan)
        return 0

    if args.device:
        os.environ["GLOBAL_CPP_DEVICE"] = args.device

    run_id = args.run_id or make_run_id()
    json_log_path = configure_structured_logging(args.log_dir, run_id, args.log_level, console=True)
    log_event(
        LOGGER, "pipeline_started", "流水线启动",
        version=LOCAL_VERSION, command=args.command, time_count=len(times), plan=plan,
        time_workers=args.time_workers, satellite_workers=args.satellite_workers,
    )

    if not args.no_preflight:
        from Preflight import run_preflight
        all_satellites = tuple(dict.fromkeys(name for item in plan for name in item["satellites"]))
        problems = run_preflight(all_satellites)
        if problems:
            log_event(
                LOGGER, "preflight_failed", "运行前检查未通过",
                level=logging.ERROR, problems=problems,
            )
            print("运行前检查未通过：", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            log_event(
                LOGGER, "pipeline_finished", "流水线因预检查失败而结束",
                level=logging.ERROR, status="PREFLIGHT_ERROR",
            )
            return 3
        log_event(LOGGER, "preflight_passed", "运行前检查通过")

    payloads = []
    for item in plan:
        payloads.append({
            **item,
            "satellite_workers": args.satellite_workers,
            "skip_retrieval": getattr(args, "skip_retrieval", False),
            "retrieval_policy": args.retrieval_policy,
            "composite_policy": args.composite_policy,
            "run_id": run_id,
            "log_dir": str(Path(args.log_dir).expanduser().resolve()) if args.log_dir else str(PROJECT_ROOT / "Logs"),
            "log_level": args.log_level,
        })
    worker = _run_satellite_payload if args.command == "satellite" else _run_global_payload
    results = execute_payloads(payloads, worker, args.time_workers)
    summary_path = write_summary(results, args.summary)
    success = all(result["status"] == "SUCCESS" for result in results)
    log_event(
        LOGGER, "pipeline_finished", "流水线结束",
        level=logging.INFO if success else logging.ERROR,
        status="SUCCESS" if success else "ERROR", results=results,
        summary_path=summary_path, first_log_path=json_log_path,
    )
    for result in results:
        print(f"{result['run_time']} | {result['product']} | {result['status']}")
    print(f"执行摘要: {summary_path}")
    print(f"结构化日志目录: {Path(json_log_path).parent}")
    return 0 if success else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
