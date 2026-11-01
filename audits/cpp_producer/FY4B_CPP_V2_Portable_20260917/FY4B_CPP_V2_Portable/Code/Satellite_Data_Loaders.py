"""
Description: 各类静止卫星亮温数据读取、定标和统一网格映射工具。
Author: Yangzx
LastEditTime: 2026-07-17
Version:
    V1.1_122+local.1
Change:
    1. 提供 GOES、FY 和 SEVIRI 亮温数据读取、定标与统一网格映射功能。
    2. local.1: 修复 GOES ``**`` 路径未启用 recursive 的问题；增加输入文件
                选择日志、映射索引和输出形状校验。
    3. local.1: 补充三类传感器通道、定标步骤和统一输出契约注释；逻辑不变。
"""

import glob
import os

import netCDF4 as nc
import numpy as np
from Structured_Logging import get_logger, log_event

LOGGER = get_logger("global_cpp.loader")

# 顺序必须与训练时输入通道顺序一致；SAZ 由各加载函数追加为第 6 通道。
GOES_Channels = ("C10", "C11", "C13", "C15", "C16")
SEVIRI_Channels = ("WV_073", "IR_087", "IR_108", "IR_120", "IR_134")


def Find_First_Matching_File(Pattern: str) -> str:
    """按文件名排序选择一个输入文件并记录选择结果。

    ``Pattern`` 可包含 ``**``；出现 ``**`` 时自动启用递归匹配。排序保证相同
    数据目录在重复运行时选择同一文件。无匹配项时抛出 ``FileNotFoundError``，
    不返回空字符串或静默跳过，以便上层触发卫星回退。
    """
    Matches = sorted(glob.glob(Pattern, recursive=("**" in Pattern)))
    if not Matches:
        raise FileNotFoundError(f"未找到文件: {Pattern}")
    Selected = Matches[0]
    log_event(
        LOGGER, "satellite_input_selected", "选择卫星输入文件",
        level=10, pattern=Pattern, selected_path=Selected, match_count=len(Matches),
    )
    return Selected


def _Validate_Mapping(LAT_Index, LON_Index, SAZ):
    """验证传感器原网格到 4051×4051 统一网格的映射输入。

    ``LAT_Index`` 和 ``LON_Index`` 是整型高级索引，``SAZ`` 是太阳天顶角及视场
    掩膜载体；三者必须同为 4051×4051。这里只检查形状，索引是否越界由实际
    传感器数组索引阶段继续验证。
    """
    if np.asarray(LAT_Index).shape != (4051, 4051):
        raise ValueError(f"LAT 索引形状异常: {np.asarray(LAT_Index).shape}")
    if np.asarray(LON_Index).shape != (4051, 4051):
        raise ValueError(f"LON 索引形状异常: {np.asarray(LON_Index).shape}")
    if np.asarray(SAZ).shape != (4051, 4051):
        raise ValueError(f"SAZ 形状异常: {np.asarray(SAZ).shape}")


def Mask_GOES_Radiance(Dataset, Radiance: np.ndarray) -> np.ndarray:
    """依据 GOES Rad 变量的填充值和有效范围屏蔽坏辐亮度。

    返回输入副本，等于 ``_FillValue``、小于等于有效下限或大于等于有效上限的
    像元设为 NaN。原数组不被修改；DQF 质量标志由后续函数独立处理。
    """
    Fill_Value = Dataset.variables["Rad"].getncattr("_FillValue")
    Valid_Range = Dataset.variables["Rad"].getncattr("valid_range")
    Masked = Radiance.copy()
    Masked[
        (Masked == Fill_Value)
        | (Masked <= Valid_Range[0])
        | (Masked >= Valid_Range[1])
    ] = np.nan
    return Masked


def Convert_GOES_BT(Dataset, Radiance: np.ndarray) -> np.ndarray:
    """使用 ABI 文件内 Planck 系数把辐亮度转换为亮温。

    先调用 :func:`Mask_GOES_Radiance` 处理填充值和有效范围，再读取 ``fk1``、
    ``fk2``、``bc1``、``bc2`` 系数执行官方反演公式。无效辐亮度在结果中保持
    NaN，返回数组形状与输入一致。
    """
    Masked = Mask_GOES_Radiance(Dataset, Radiance)
    FK1 = Dataset.variables["planck_fk1"][:]
    FK2 = Dataset.variables["planck_fk2"][:]
    BC1 = Dataset.variables["planck_bc1"][:]
    BC2 = Dataset.variables["planck_bc2"][:]
    return (FK2 / (np.log((FK1 / Masked) + 1)) - BC1) / BC2


def _Read_GOES_DQF_Bad_Mask(Dataset, Shape):
    """读取 GOES DQF 并返回坏质量布尔掩膜。

    参考 GEO 规则，只有 DQF=0 视为可用，其他值和掩膜填充值均为坏质量。
    DQF 缺失时记录警告并返回全 False，以兼容历史文件；存在但形状与 Rad 不同
    时抛出 ``ValueError``，避免错误广播。
    """
    if "DQF" not in Dataset.variables:
        log_event(
            LOGGER, "goes_dqf_missing", "GOES 文件缺少 DQF，继续使用辐亮度有效范围质控",
            level=30,
        )
        return np.zeros(Shape, dtype=bool)
    DQF = np.asarray(np.ma.filled(Dataset.variables["DQF"][:], -1))
    if DQF.shape != Shape:
        raise ValueError(f"GOES DQF 形状异常: {DQF.shape}，预期 {Shape}")
    return DQF != 0


def Load_GOES_BT(
    Path_Dir: str,
    Date_Str: str,
    GOES_ID: str,
    LAT_Index: np.ndarray,
    LON_Index: np.ndarray,
    SAZ: np.ndarray,
) -> np.ndarray:
    """读取 GOES ABI 五个红外通道并映射到统一网格。

    依次读取 C10/C11/C13/C15/C16，优先搜索新版年度目录，找不到时回退旧归档
    路径。每通道完成辐亮度到亮温转换和 DQF 检查，再用 ``LAT_Index/LON_Index``
    映射到 4051×4051；SAZ 为 NaN 的视场外像元和任一通道 DQF 非 0 的像元均设
    NaN。最后追加 SAZ 为第六通道并翻转纬度，返回 ``[6,4051,4051]`` float 数组。
    """
    _Validate_Mapping(LAT_Index, LON_Index, SAZ)
    Year = Date_Str[:4]
    Year_Dir = f"data_goes{GOES_ID}_{Year}0010000_{Year}3652300"
    # 新路径 (2024+): /GOES/GOES-{ID}/{Year}/**/ 下递归搜索
    New_Base = os.path.join(Path_Dir, f"GOES-{GOES_ID}", Year)
    BT_List = []
    DQF_Bad_List = []

    for Channel in GOES_Channels:
        # 先试新路径, 回退旧路径
        New_Pat = os.path.join(New_Base, "**", f"OR_ABI-L1b-RadF-M*{Channel}_G{GOES_ID}_s{Date_Str}*.nc")
        Old_Pat = os.path.join(Path_Dir, Year_Dir, f"OR_ABI-L1b-RadF-M*{Channel}_G{GOES_ID}_s{Date_Str}*.nc")
        try:
            Path = Find_First_Matching_File(New_Pat)
        except FileNotFoundError:
            Path = Find_First_Matching_File(Old_Pat)
        with nc.Dataset(Path) as Dataset:
            Radiance = Dataset.variables["Rad"][:, :]
            BT_List.append(Convert_GOES_BT(Dataset, Radiance))
            DQF_Bad_List.append(_Read_GOES_DQF_Bad_Mask(Dataset, Radiance.shape))

    BT = np.asarray(BT_List)
    DQF_Bad = np.asarray(DQF_Bad_List)
    Func_BT = BT[:, LAT_Index, LON_Index]
    Func_DQF_Bad = DQF_Bad[:, LAT_Index, LON_Index]
    Func_BT[:, np.isnan(SAZ)] = np.nan
    Func_BT[Func_DQF_Bad] = np.nan
    GOES_Data = np.vstack([Func_BT, SAZ.reshape(1, 4051, 4051)])
    return GOES_Data[:, ::-1, :]


def Calibrate_FY_Channel(Raw_Channel: np.ndarray, Calibration: np.ndarray) -> np.ndarray:
    """使用 FY AGRI 定标查找表将原始计数转换为亮温。

    只有落在 ``[0, len(Calibration))`` 的整数计数才参与查表，负值和越界值保留
    为 NaN。返回 float32 数组且不修改原始通道。
    """
    Channel = Raw_Channel.astype(np.int32)
    Calibration = np.asarray(Calibration)

    BT = np.full(Channel.shape, np.nan, dtype=np.float32)

    Valid_Mask = (Channel >= 0) & (Channel < len(Calibration))
    BT[Valid_Mask] = Calibration[Channel[Valid_Mask]]

    return BT


def Load_FY4_AGRI_BT(
    Path: str,
    LAT_Index: np.ndarray,
    LON_Index: np.ndarray,
    SAZ: np.ndarray,
    Channel_Numbers,
    Data_Group: str = None,
    Calibration_Group: str = None,
) -> np.ndarray:
    """读取 FY-4 AGRI 指定通道、完成定标并映射到统一网格。

    ``Channel_Numbers`` 的顺序必须与对应模型训练通道一致；可通过
    ``Data_Group`` 和 ``Calibration_Group`` 兼容不同 FY 文件内部层级。每通道
    使用定标表转换后按映射索引抽取，SAZ 为 NaN 的位置统一屏蔽。返回五个亮温
    通道加 SAZ 的 ``[6,4051,4051]`` 数组。
    """
    _Validate_Mapping(LAT_Index, LON_Index, SAZ)
    BT_List = []
    with nc.Dataset(Path) as Dataset:
        Data_Root = Dataset[Data_Group] if Data_Group else Dataset
        Calibration_Root = Dataset[Calibration_Group] if Calibration_Group else Dataset

        for Channel in Channel_Numbers:
            Raw_Channel = Data_Root[f"NOMChannel{Channel}"][:]
            Calibration = Calibration_Root[f"CALChannel{Channel}"][:]
            BT_List.append(Calibrate_FY_Channel(Raw_Channel, Calibration))

    BT = np.asarray(BT_List)
    Func_BT = BT[:, LAT_Index, LON_Index]
    Func_BT[:, np.isnan(SAZ)] = np.nan
    return np.vstack([Func_BT, SAZ.reshape(1, 4051, 4051)])


def Load_SEVIRI_BT(
    Path: str,
    LAT_Index: np.ndarray,
    LON_Index: np.ndarray,
    SAZ: np.ndarray,
) -> np.ndarray:
    """使用 Satpy 读取 SEVIRI 五通道亮温并映射到统一网格。

    按训练顺序加载 WV_073、IR_087、IR_108、IR_120、IR_134。Satpy 输出先在
    两个空间轴翻转以匹配历史网格方向，再按映射索引抽取；SAZ 为 NaN 的视场外
    像元设为 NaN。未安装 Satpy 时给出明确依赖错误。返回形状为
    ``[6,4051,4051]``，最后一个通道为 SAZ。
    """
    _Validate_Mapping(LAT_Index, LON_Index, SAZ)
    import xarray as xr
    try:
        from satpy.scene import Scene
    except ImportError as Exc:
        raise ImportError("读取 SEVIRI 数据需要先安装 satpy。") from Exc

    Scene_Obj = Scene(filenames=[Path], reader="seviri_l1b_native")
    Scene_Obj.load(list(SEVIRI_Channels))

    BT_List = []
    for Channel in SEVIRI_Channels:
        SAT_Data = Scene_Obj[Channel].values[:, :]
        SAT_Array = xr.DataArray(SAT_Data, name="Brightness_Temperature")
        BT_List.append(SAT_Array[::-1, ::-1].values)

    BT = np.asarray(BT_List)
    Func_BT = BT[:, LAT_Index, LON_Index]
    Func_BT[:, np.isnan(SAZ)] = np.nan
    return np.vstack([Func_BT, SAZ.reshape(1, 4051, 4051)])
