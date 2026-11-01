"""
Description: 匹配 ERA 温湿廓线、地表温度和地表类型，并重采样到卫星统一网格。
Author: Yangzx
LastEditTime: 2026-07-17
Version:
    V1.1_122+local.1
Change:
    1. 提供 ERA 温湿廓线、地表温度和地表类型的读取、裁剪与重采样功能。
    2. local.1: ERA5 与 LCCS 根目录支持环境变量覆盖。
    3. local.1: 所有 xarray Dataset 使用上下文管理器关闭；重采样使用
                inference_mode，避免长时间序列积累文件句柄和计算图元数据。
    4. local.1: 细化变量、通道、经度跨界和返回形状注释；处理逻辑不变。
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from Structured_Logging import get_logger, log_event

LOGGER = get_logger("global_cpp.auxiliary")

ERA5_Legacy_Root = os.getenv('GLOBAL_CPP_ERA5_ROOT', '/home/Data_Pool/data/ERA5')
ERA5_Product_Root = os.getenv('GLOBAL_CPP_ERA5_PRODUCT_ROOT', '/home/Data_Pool_3/data/Global_CPP/ERA5')
ATP_Dir = [f'{ERA5_Legacy_Root}/profile/T', f'{ERA5_Product_Root}/profile/T']
RHP_Dir = [f'{ERA5_Legacy_Root}/profile/R', f'{ERA5_Product_Root}/profile/R']
SKT_Dir = f'{ERA5_Product_Root}/single/SKT'
LCCS_Dir = os.getenv('GLOBAL_CPP_LCCS_ROOT', '/home/Data_Pool_3/data/Global_CPP/LCCS/MCD12_IGBP_CMG')


def Convert_High_Resolution(Input_Data):
    """把低分辨率辅助场双线性重采样到单星 4051×4051 网格。"""
    Input_Array = np.asarray(Input_Data)
    if Input_Array.ndim != 3:
        raise ValueError(f"重采样输入应为 [C,H,W]，实际 {Input_Array.shape}")
    Input_Tensor = torch.as_tensor(Input_Array, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        Output = F.interpolate(
            Input_Tensor, size=(4051, 4051), mode='bilinear', align_corners=True
        )
    return Output.numpy()[0]


def Convert_Categorical_High_Resolution(Input_Data):
    """将离散分类场最近邻重采样到 ``[C,4051,4051]``。

    LCCS 是 IGBP 类别编号，不能使用双线性插值，否则会在类别边界产生不存在的
    小数类别。部分跨日界线区域因 NetCDF 端点同时包含会得到 4052 列，本函数
    同时消除该一列差异，并保持原类别值不被混合。
    """
    Input_Array = np.asarray(Input_Data)
    if Input_Array.ndim != 3:
        raise ValueError(f"分类场重采样输入应为 [C,H,W]，实际 {Input_Array.shape}")
    Input_Tensor = torch.as_tensor(Input_Array, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        Output = F.interpolate(Input_Tensor, size=(4051, 4051), mode='nearest')
    return Output.numpy()[0]


def Get_ERA_LCCS(Run_Time, Border_N, Border_S, Border_E, Border_W):
    """读取并组合模型使用的 10 通道环境辅助场。

    返回 ``[10,4051,4051]``：4 层气温、4 层相对湿度、1 层地表温度和
    1 层 IGBP 地表类型。各卫星适配器可通过 ``ERA_Transform`` 调整前 8 通道
    顺序，但不得改变通道总数。ERA 时次按整点匹配。
    """
    ATP = Get_ATP(Run_Time, Border_N, Border_S, Border_E, Border_W)
    RHP = Get_RHP(Run_Time, Border_N, Border_S, Border_E, Border_W)
    SKT = Get_SKT(Run_Time, Border_N, Border_S, Border_E, Border_W)

    LCCS = Get_LCCS(Run_Time, Border_N, Border_S, Border_E, Border_W)
    LCCS = Convert_Categorical_High_Resolution(np.expand_dims(LCCS, axis=0))
    return np.concatenate([ATP, RHP, SKT, LCCS], axis=0)


def Get_ATP(Run_Time, Border_N, Border_S, Border_E, Border_W):
    """读取 300/500/850/1000 hPa 气温并重采样为 ``[4,4051,4051]``。"""
    ATP_Name = Run_Time.strftime('era5-temperature-%Y%m%d')
    if int(Run_Time.year) > 2023:
        ATP_Path = f'{ATP_Dir[1]}/{Run_Time.year}/{ATP_Name}.nc'
    else:
        ATP_Path = f'{ATP_Dir[0]}/{Run_Time.year}/{ATP_Name}.nc'

    log_event(LOGGER, "era_temperature_opened", "读取 ERA5 温度廓线", level=10, path=ATP_Path)
    with xr.open_dataset(ATP_Path, engine='netcdf4') as ATP_Dataset:
        Time_Var = 'valid_time' if 'valid_time' in ATP_Dataset.coords else 'time'
        Level_Var = 'pressure_level' if 'pressure_level' in ATP_Dataset.coords else 'level'
        ATP_Selected = ATP_Dataset.sel(
            **{Time_Var: Run_Time.replace(minute=0)},
            **{Level_Var: [300, 500, 850, 1000]},
            latitude=slice(Border_N, Border_S)
        )
        West_360, East_360 = Border_W % 360, Border_E % 360
        if West_360 <= East_360:
            ATP_Selected = ATP_Selected.sel(longitude=slice(West_360, East_360))
        else:
            ATP_Selected = xr.concat([
                ATP_Selected.sel(longitude=slice(West_360, 360)),
                ATP_Selected.sel(longitude=slice(0, East_360)),
            ], dim='longitude')
        ATP = ATP_Selected.t.values
    return Convert_High_Resolution(ATP)


def Get_RHP(Run_Time, Border_N, Border_S, Border_E, Border_W):
    """读取四个指定气压层相对湿度并重采样为 ``[4,4051,4051]``。"""
    RHP_Name = Run_Time.strftime('era5-relative_humidity-%Y%m%d')
    if int(Run_Time.year) > 2023:
        RHP_Path = f'{RHP_Dir[1]}/{Run_Time.year}/{RHP_Name}.nc'
    else:
        RHP_Path = f'{RHP_Dir[0]}/{Run_Time.year}/{RHP_Name}.nc'

    log_event(LOGGER, "era_humidity_opened", "读取 ERA5 相对湿度廓线", level=10, path=RHP_Path)
    with xr.open_dataset(RHP_Path, engine='netcdf4') as RHP_Dataset:
        Time_Var = 'valid_time' if 'valid_time' in RHP_Dataset.coords else 'time'
        Level_Var = 'pressure_level' if 'pressure_level' in RHP_Dataset.coords else 'level'
        RHP_Selected = RHP_Dataset.sel(
            **{Time_Var: Run_Time.replace(minute=0)},
            **{Level_Var: [300, 500, 850, 1000]},
            latitude=slice(Border_N, Border_S)
        )
        West_360, East_360 = Border_W % 360, Border_E % 360
        if West_360 <= East_360:
            RHP_Selected = RHP_Selected.sel(longitude=slice(West_360, East_360))
        else:
            RHP_Selected = xr.concat([
                RHP_Selected.sel(longitude=slice(West_360, 360)),
                RHP_Selected.sel(longitude=slice(0, East_360)),
            ], dim='longitude')
        RHP = RHP_Selected.r.values
    return Convert_High_Resolution(RHP)


def Get_SKT(Run_Time, Border_N, Border_S, Border_E, Border_W):
    """读取整点 ERA5 skin temperature，返回 ``[1,4051,4051]``。"""
    SKT_Name = Run_Time.strftime('era5-skin_temperature-%Y%m%d')
    SKT_Path = f'{SKT_Dir}/{Run_Time.year}/{SKT_Name}.nc'
    log_event(LOGGER, "era_skin_temperature_opened", "读取 ERA5 地表温度", level=10, path=SKT_Path)
    with xr.open_dataset(SKT_Path, engine='netcdf4') as SKT_Dataset:
        Time_Var = 'valid_time' if 'valid_time' in SKT_Dataset.coords else 'time'
        SKT_Selected = SKT_Dataset.sel(
            **{Time_Var: Run_Time.replace(minute=0)}, latitude=slice(Border_N, Border_S)
        )
        West_360, East_360 = Border_W % 360, Border_E % 360
        if West_360 <= East_360:
            SKT_Selected = SKT_Selected.sel(longitude=slice(West_360, East_360))
        else:
            SKT_Selected = xr.concat([
                SKT_Selected.sel(longitude=slice(West_360, 360)),
                SKT_Selected.sel(longitude=slice(0, East_360)),
            ], dim='longitude')
        SKT = SKT_Selected['skt'].values
    SKT = np.expand_dims(SKT, axis=0)
    return Convert_High_Resolution(SKT)


def Get_LCCS(Run_Time, Border_N, Border_S, Border_E, Border_W):
    """读取当年 MCD12 IGBP 地表类型二维子区。

    当东界大于 180° 时，分别读取日界线两侧再按经度方向拼接，避免物化完整
    全球场。该离散分类场不在此函数内插值，统一重采样由上层组合流程处理。
    """
    LCCS_Path = f'{LCCS_Dir}/MCD12_IGBP_{Run_Time.year}_Global_CMG.nc'
    log_event(LOGGER, "lccs_opened", "读取 MCD12 IGBP 地表类型", level=10, path=LCCS_Path)
    with xr.open_dataset(LCCS_Path, engine='netcdf4') as LCCS_Dataset:
        if Border_E > 180:
            # 只读取目标纬度和跨日界线的两个经度片段；旧实现先拼完整全球场，
            # 会额外分配约 4501×9001 的数组。
            LCCS_Field = LCCS_Dataset['LCCS_IGBP'].sel(LAT=slice(Border_N, Border_S))
            LCCS_W = LCCS_Field.sel(LON=slice(Border_W, 180)).values
            LCCS_E = LCCS_Field.sel(LON=slice(-180, Border_E - 360)).values
            return np.concatenate((LCCS_W, LCCS_E), axis=1)
        return LCCS_Dataset['LCCS_IGBP'].sel(
            LAT=slice(Border_N, Border_S), LON=slice(Border_W, Border_E)
        ).values
