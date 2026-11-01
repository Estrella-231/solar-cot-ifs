"""单颗静止卫星 CPP 反演的公共编排层。

模块职责：把各卫星适配器提供的亮温读取函数与统一的 ERA/LCCS 辅助场、切片、
四模型推理和原子化产品写出串联起来。卫星差异只保留在 ``Retrieval_Spec`` 中，
公共流程负责形状校验、已有结果判断、结构化日志、异常传播和内存清理。

数据契约：亮温为 ``[6,4051,4051]``（5 个红外通道+太阳天顶角），辅助场为
``[10,4051,4051]``（4 层温度、4 层相对湿度、SKT、LCCS），拼接后形成模型所需
``[16,4051,4051]``。输出包括 CPP NetCDF 与三分类分数 CLP_P NPZ；CLP_P 为
兼容历史命名保留的目录名，数值是非负模型分数，不是 0–1 softmax 概率。

版本历史：
    V1.1_122          各卫星独立执行公共反演步骤。
    V1.1_122+local.1  统一异常语义，增加输入校验、分阶段结构日志、finally 大数组
                         清理，并补充配置字段、通道顺序及失败边界注释。
"""

import os
import gc
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from datetime import timedelta

import numpy as np
from Data_Preprocess_FD import Cut
from Get_ERA_LCCS import Get_ERA_LCCS
from Retrieval_CPP import Check_Output_Pair, Retrieval, Write_To_NC_Using_NC
from Output_Policy import normalize_output_policy
from Structured_Logging import configure_structured_logging, get_logger, log_event
from Quality_Control import apply_previous_observation_fallback, classify_cloud_phase

LOGGER = get_logger("global_cpp.satellite")

BT_Loader = Callable[[object], np.ndarray]
ERA_Transformer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Retrieval_Region:
    """单星统一经纬网格边界，单位为度。

    纬度按北到南排列；经度允许使用负值表示西经。跨越 0°/180° 的具体切片
    由辅助数据读取模块处理。对象冻结以避免运行中意外修改卫星覆盖范围。
    """
    Border_N: float
    Border_S: float
    Border_E: float
    Border_W: float


@dataclass(frozen=True)
class Retrieval_Spec:
    """一颗卫星执行公共反演流程所需的不可变配置。

    ``Model_Paths`` 顺序必须是 CLP、CTH、CER、COT；``XMAXMIN`` 必须是
    ``[2,16]``，第 0 行是归一化上界、第 1 行是下界。``ERA_Transform`` 用于
    兼容训练阶段采用不同辅助通道顺序的 FY 模型。``Overwrite_Existing`` 仅控制
    是否跳过已存在且通过校验的 CPP，不会放宽写出后的完整性检查。
    """
    Name: str
    CPP_Root: str
    CLP_P_Root: str
    Output_Prefix: str
    Model_Paths: Tuple[str, str, str, str]
    XMAXMIN: np.ndarray
    Region: Retrieval_Region
    Overwrite_Existing: bool = False
    ERA_Transform: Optional[ERA_Transformer] = None


def Reorder_ERA_For_FY(ERA_Data: np.ndarray) -> np.ndarray:
    """将十通道 ERA5/LCCS 辅助场重排为 FY 模型训练顺序。

    输入默认顺序为前四层温度、后四层相对湿度、SKT、LCCS；FY 权重要求先湿度
    后温度，因此返回 ``[4:8] + [0:4] + [8:]``。只重排第一维，不改变空间网格。
    """
    return np.concatenate((ERA_Data[4:8], ERA_Data[:4], ERA_Data[8:]), axis=0)


def Build_Output_Paths(Run_Time, Spec: Retrieval_Spec) -> Tuple[str, str]:
    """根据 UTC 时次和卫星配置构造一对输出路径。

    CPP NetCDF 与 CLP_P NPZ 使用同一 ``YYYY/新YYYYmmdd`` 目录层级和秒级时间戳，
    仅根目录及扩展名不同。函数只生成字符串，不创建目录也不检查文件存在性。
    """
    Time_Str = Run_Time.strftime("%Y%m%d%H%M%S")
    Year = Time_Str[:4]
    Day = Time_Str[:8]
    CPP_Path = f"{Spec.CPP_Root}/{Year}/{Day}/{Spec.Output_Prefix}_{Time_Str}.nc"
    CLP_P_Path = f"{Spec.CLP_P_Root}/{Year}/{Day}/{Spec.Output_Prefix}_{Time_Str}.npz"
    return CPP_Path, CLP_P_Path


def _Read_Previous_Output(Run_Time, Spec: Retrieval_Spec):
    """读取上一小时同一卫星已发布的 CPP/CLP_P；缺失时返回空回退源。"""
    from netCDF4 import Dataset
    Previous_CPP, Previous_CLPP = Build_Output_Paths(Run_Time - timedelta(hours=1), Spec)
    if not (os.path.isfile(Previous_CPP) and os.path.isfile(Previous_CLPP)):
        return None, None
    try:
        with Dataset(Previous_CPP) as ds, np.load(Previous_CLPP) as scores_file:
            values = {name: np.asarray(ds.variables[name][:], dtype=np.float32) for name in ("CLP", "CTH", "CER", "COT")}
            scores = np.asarray(scores_file["arr_0"], dtype=np.float32)
        return values, scores
    except Exception as exc:
        log_event(LOGGER, "previous_satellite_output_unreadable", "上一小时单星结果无法作为回填源", level=logging.WARNING, satellite=Spec.Name, run_time=Run_Time, error=repr(exc))
        return None, None


def _Build_Observation_Invalid_Mask(BT: np.ndarray) -> np.ndarray:
    """从加载器已应用的原始质量标志生成二维观测无效掩膜。

    加载器把 GOES DQF、辐亮度填充值、FY 无效计数/定标和 SEVIRI 掩膜值统一转换
    为 NaN；SAZ 的 NaN 表示卫星视场外。因此任一红外通道或 SAZ 非有限即为
    观测异常，而非依据模型预测数值判断。
    """
    return np.any(~np.isfinite(BT[:5]), axis=0) | ~np.isfinite(BT[5])


def Create_Empty_Outputs():
    """构造与正式模型输出契约一致的全 NaN 结果。

    仅当环境变量显式允许错误时写空产品时使用。返回三通道 CLP 分数和三个
    4051×4051 连续产品；默认生产流程会传播异常而不会调用此回退。
    """
    return (
        np.full((3, 4051, 4051), np.nan),
        np.full((4051, 4051), np.nan),
        np.full((4051, 4051), np.nan),
        np.full((4051, 4051), np.nan),
    )


def _Validate_Input_Array(Data: np.ndarray, Expected_Channels: int, Name: str) -> None:
    """验证卫星或辅助场输入的类型、通道数与空间尺寸。

    输入必须是 NumPy 三维数组 ``[channel,4051,4051]``，通道数由调用方指定。
    失败时抛出带数据名称和实际形状的异常，使错误停留在数据边界，而不是延后到
    NumPy 拼接或模型卷积阶段。
    """
    if not isinstance(Data, np.ndarray):
        raise TypeError(f"{Name} 必须是 numpy.ndarray，实际为 {type(Data).__name__}")
    if Data.ndim != 3:
        raise ValueError(f"{Name} 应为 3 维 [C,H,W]，实际形状 {Data.shape}")
    if Data.shape[0] != Expected_Channels:
        raise ValueError(f"{Name} 应有 {Expected_Channels} 个通道，实际 {Data.shape[0]}")
    if Data.shape[1:] != (4051, 4051):
        raise ValueError(f"{Name} 网格应为 4051×4051，实际 {Data.shape[1:]}")


def Run_Satellite_Retrieval(Run_Time, Spec: Retrieval_Spec, Load_BT: BT_Loader) -> None:
    """执行一颗卫星一个时次的完整反演与原子写出。

    流程依次为：构造输出路径 → 按输出策略判断跳过/覆盖/修复 → 读取五通道亮温
    和 SAZ → 读取 ERA5/LCCS → 组合 16 通道输入 → 惰性切成 tile → 四模型推理 →
    写出 CPP 与 CLP_P。每个阶段均记录结构化日志和耗时。

    ``Spec`` 提供模型、归一化参数、覆盖范围和输出目录；``Load_BT`` 隔离具体
    传感器读取差异。默认情况下任何读取、推理或写出异常都会向上抛出，以便全球
    拼接层尝试上一时次结果。finally 块解除 GB 级数组引用并执行垃圾回收，保证
    多时次运行不会因对象累积耗尽内存。
    """
    configure_structured_logging(console=True)
    CPP_Path, CLP_P_Path = Build_Output_Paths(Run_Time, Spec)
    Retrieval_Policy = normalize_output_policy(
        os.getenv("GLOBAL_CPP_RETRIEVAL_POLICY", "skip")
    )
    if Retrieval_Policy == "overwrite":
        log_event(
            LOGGER, "satellite_overwrite_enabled", "当前卫星配置要求重新反演并原子替换已有结果",
            satellite=Spec.Name, run_time=Run_Time, cpp_path=CPP_Path,
            policy=Retrieval_Policy,
        )
    elif Check_Output_Pair(CPP_Path, CLP_P_Path, Retrieval_Policy):
        log_event(
            LOGGER, "satellite_output_skipped", "已有单星产品通过校验，跳过反演",
            satellite=Spec.Name, run_time=Run_Time, cpp_path=CPP_Path,
            policy=Retrieval_Policy,
        )
        return

    BT = None
    Observation_Invalid = None
    QA = None
    ERA = None
    Input_Tiles = None
    Per_CPP = None
    started = time.perf_counter()
    
    try:
        stage_started = time.perf_counter()
        BT = Load_BT(Run_Time)
        if BT is None:
            raise ValueError("卫星亮温数据为空")
        _Validate_Input_Array(BT, 6, "卫星输入")
        log_event(
            LOGGER, "satellite_input_loaded", "卫星亮温与观测几何读取完成",
            satellite=Spec.Name, run_time=Run_Time, shape=BT.shape,
            elapsed_seconds=round(time.perf_counter() - stage_started, 3),
        )

        stage_started = time.perf_counter()
        ERA = Get_ERA_LCCS(
            Run_Time,
            Spec.Region.Border_N,
            Spec.Region.Border_S,
            Spec.Region.Border_E,
            Spec.Region.Border_W,
        )
        if Spec.ERA_Transform is not None:
            ERA = Spec.ERA_Transform(ERA)
        _Validate_Input_Array(ERA, 10, "ERA5/LCCS 输入")
        log_event(
            LOGGER, "era_lccs_loaded", "ERA5/LCCS 辅助场读取完成",
            satellite=Spec.Name, run_time=Run_Time, shape=ERA.shape,
            elapsed_seconds=round(time.perf_counter() - stage_started, 3),
        )

        Combined = np.concatenate((BT, ERA), axis=0).astype(np.float32, copy=False)
        _Validate_Input_Array(Combined, 16, "模型输入")
        Observation_Invalid = _Build_Observation_Invalid_Mask(BT)
        Input_Tiles = Cut(Combined)
        del Combined
        BT = ERA = None
        gc.collect()
        log_event(
            LOGGER, "input_tiles_built", "模型输入切片完成",
            satellite=Spec.Name, run_time=Run_Time, tile_shape=Input_Tiles.shape,
        )

        Per_CPP = Retrieval(Input_Tiles, list(Spec.Model_Paths), Spec.XMAXMIN)
        Current_Values = {"CLP": classify_cloud_phase(Per_CPP[0]), "CTH": Per_CPP[1], "CER": Per_CPP[2], "COT": Per_CPP[3]}
        Previous_Values, Previous_CLPP = _Read_Previous_Output(Run_Time, Spec)
        Updated_Values, Updated_CLPP, QA, Fallback_Stats = apply_previous_observation_fallback(
            Current_Values, Per_CPP[0], Observation_Invalid, Previous_Values, Previous_CLPP
        )
        Per_CPP = (Updated_CLPP, Updated_Values["CTH"], Updated_Values["CER"], Updated_Values["COT"])
        log_event(LOGGER, "satellite_observation_fallback_applied", "已按原始观测质量标志执行单星上一时次回填", satellite=Spec.Name, run_time=Run_Time, **Fallback_Stats)
    except Exception as Exc:
        log_event(
            LOGGER, "satellite_pipeline_failed", "单星公共流程失败",
            level=logging.ERROR, exc_info=True,
            satellite=Spec.Name, run_time=Run_Time,
            elapsed_seconds=round(time.perf_counter() - started, 3), error=repr(Exc),
        )
        # 整景原始文件缺失、单个通道缺失或推理错误时，也要把上一小时“已写出
        # 的结果”作为当前时次输出。这样连续缺测的第二小时会引用第一小时已填补
        # 的文件，而不是让全球拼接只能寻找不存在的原始当前文件。
        Previous_Values, Previous_CLPP = _Read_Previous_Output(Run_Time, Spec)
        if Previous_Values is not None and Previous_CLPP is not None:
            Empty_CPP = Create_Empty_Outputs()
            Empty_Values = {
                "CLP": classify_cloud_phase(Empty_CPP[0]),
                "CTH": Empty_CPP[1], "CER": Empty_CPP[2], "COT": Empty_CPP[3],
            }
            Observation_Invalid = np.ones((4051, 4051), dtype=bool)
            Updated_Values, Updated_CLPP, QA, Fallback_Stats = apply_previous_observation_fallback(
                Empty_Values, Empty_CPP[0], Observation_Invalid, Previous_Values, Previous_CLPP
            )
            if np.any(QA):
                Per_CPP = (
                    Updated_CLPP, Updated_Values["CTH"], Updated_Values["CER"], Updated_Values["COT"],
                )
                log_event(
                    LOGGER, "satellite_whole_scene_fallback_applied",
                    "当前整景不可用，已使用上一小时单星结果链式回填",
                    level=logging.WARNING, satellite=Spec.Name, run_time=Run_Time,
                    original_error=repr(Exc), **Fallback_Stats,
                )
            else:
                Previous_Values = Previous_CLPP = None
        if Per_CPP is None:
            if os.getenv("GLOBAL_CPP_WRITE_EMPTY_ON_ERROR", "0") != "1":
                BT = Observation_Invalid = QA = ERA = Input_Tiles = Per_CPP = None
                gc.collect()
                raise RuntimeError(f"{Run_Time} {Spec.Name} 单星反演失败") from Exc
            Per_CPP = Create_Empty_Outputs()
            Observation_Invalid = np.ones((4051, 4051), dtype=bool)
            QA = np.zeros((4051, 4051), dtype=bool)

    try:
        os.makedirs(os.path.dirname(CPP_Path), exist_ok=True)
        os.makedirs(os.path.dirname(CLP_P_Path), exist_ok=True)
        Write_To_NC_Using_NC(
            Per_CPP,
            Observation_Invalid,
            QA,
            CPP_Path,
            CLP_P_Path,
            Spec.Region.Border_N,
            Spec.Region.Border_S,
            Spec.Region.Border_E,
            Spec.Region.Border_W,
        )
        log_event(
            LOGGER, "satellite_pipeline_finished", "单星公共流程完成",
            satellite=Spec.Name, run_time=Run_Time,
            cpp_path=CPP_Path, clp_probability_path=CLP_P_Path,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        # 长时间序列中异常也必须走到这里，显式解除对 GB 级数组的引用。
        BT = Observation_Invalid = QA = ERA = Input_Tiles = Per_CPP = None
        gc.collect()
