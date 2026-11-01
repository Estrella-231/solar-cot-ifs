"""静止卫星随时间更替的唯一配置源。

模块职责：集中维护生产全球产品所需的 FY、GOES 东/西和 MSG 0°/IODC 观测
组合。统一入口的计划生成、活动卫星校验和全球拼接前置反演都必须查询本模块，
避免日期边界散落在多个脚本中并产生不一致。

区间语义：全部使用左闭右开 ``[start, end)``。服务器 V1.1_122 在
2024-03-01 00:00 存在规则空档，本地版明确该时次继续采用 FY4B_133E，直到
2024-03-01 01:00 才切换到 FY4B_105E。这里的名称表示“卫星+观测槽位”，因此
同一物理卫星换经度后也作为不同配置处理。

版本历史：
    V1.1_122+local.1  建立唯一时间线并补齐 2024-03-01 00:00 空档；细化区间、
                         名称与校验语义注释。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple


KNOWN_SATELLITES: Tuple[str, ...] = (
    "FY4A_105E", "FY4B_105E", "FY4B_133E",
    "GOES16", "GOES17", "GOES18",
    "MSG1_IODC", "MSG2_IODC", "MSG3_0DEG", "MSG4_0DEG",
)


@dataclass(frozen=True)
class Satellite_Period:
    """一个半开生产时段及其全球拼接卫星组合。

    ``start=None`` 表示无下界，``end=None`` 表示无上界。``satellites`` 的顺序
    是稳定的调度顺序；``note`` 只供计划展示和维护说明，不参与条件判断。
    """

    name: str
    start: Optional[datetime]
    end: Optional[datetime]
    satellites: Tuple[str, ...]
    note: str

    def contains(self, run_time: datetime) -> bool:
        """判断时次是否落入本区间，结束时刻严格归入下一时段。"""
        return ((self.start is None or run_time >= self.start)
                and (self.end is None or run_time < self.end))


SATELLITE_PERIODS: Tuple[Satellite_Period, ...] = (
    Satellite_Period(
        "P1_FY4A_GOES17_MSG1_MSG4", None, datetime(2022, 6, 1, 9),
        ("FY4A_105E", "GOES16", "GOES17", "MSG1_IODC", "MSG4_0DEG"),
        "FY-4A、GOES-17、MSG-1/4 组合",
    ),
    Satellite_Period(
        "P2_FY4B_GOES17_MSG2_MSG4", datetime(2022, 6, 1, 9), datetime(2023, 1, 1),
        ("FY4B_133E", "GOES16", "GOES17", "MSG2_IODC", "MSG4_0DEG"),
        "FY-4B 133E 与 MSG-2 接替",
    ),
    Satellite_Period(
        "P3_FY4B_GOES18_MSG2_MSG4", datetime(2023, 1, 1), datetime(2023, 3, 21),
        ("FY4B_133E", "GOES16", "GOES18", "MSG2_IODC", "MSG4_0DEG"),
        "GOES-18 接替 GOES-17",
    ),
    Satellite_Period(
        "P4_FY4B_GOES18_MSG2_MSG3", datetime(2023, 3, 21), datetime(2024, 3, 1, 1),
        ("FY4B_133E", "GOES16", "GOES18", "MSG2_IODC", "MSG3_0DEG"),
        "MSG-3 接替 MSG-4；本地版将区间延续至 2024-03-01 01:00",
    ),
    Satellite_Period(
        "P5_FY4B105_GOES18_MSG2_MSG3", datetime(2024, 3, 1, 1), None,
        ("FY4B_105E", "GOES16", "GOES18", "MSG2_IODC", "MSG3_0DEG"),
        "FY-4B 观测位置切换为 105E",
    ),
)


def get_period(run_time: datetime) -> Satellite_Period:
    """返回时次所属区间；配置有空档时抛出 ``ValueError``。"""
    for period in SATELLITE_PERIODS:
        if period.contains(run_time):
            return period
    raise ValueError(f"未找到 {run_time!s} 对应的卫星组合")


def get_active_satellites(run_time: datetime) -> Tuple[str, ...]:
    """返回生产全球产品时应参与的五颗卫星/观测配置。"""
    return get_period(run_time).satellites


def validate_satellites(names: Sequence[str]) -> Tuple[str, ...]:
    """转为大写、去重且保序，并校验用户显式指定的卫星名称。"""
    normalized = tuple(dict.fromkeys(name.strip().upper() for name in names))
    unknown = sorted(set(normalized) - set(KNOWN_SATELLITES))
    if unknown:
        raise ValueError(f"未知卫星: {', '.join(unknown)}")
    if not normalized:
        raise ValueError("至少需要指定一颗卫星")
    return normalized


def inactive_satellites(run_time: datetime, names: Sequence[str]) -> Tuple[str, ...]:
    """返回显式选择中不属于当前生产组合的卫星。"""
    active = set(get_active_satellites(run_time))
    return tuple(name for name in names if name not in active)
