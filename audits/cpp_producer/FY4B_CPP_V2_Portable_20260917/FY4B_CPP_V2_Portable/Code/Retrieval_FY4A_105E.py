"""FY-4A 105°E AGRI 单星适配器。

读取 AGRI 10–14 号红外通道并完成定标，叠加太阳天顶角后输出
``[6,4051,4051]``；公共流程再拼接 10 通道 ERA/LCCS，供 16 通道模型推理。
统一覆盖区为 81°N–81°S、24°E–186°E。原始数据根目录可由
``GLOBAL_CPP_FY_ROOT`` 覆盖，映射、模型及输出路径由 ``Project_Paths`` 管理。

版本历史：V1.1 延迟读取 SAZ NetCDF，避免导入期累积 HDF5 句柄；local.1
加入可配置路径及通道、形状、区域和路径说明，反演逻辑不变。
"""

import os
import sys
from pathlib import Path

Project_Root_For_Import = Path(__file__).resolve().parent.parent
if str(Project_Root_For_Import) not in sys.path:
    sys.path.insert(0, str(Project_Root_For_Import))

from datetime import datetime

import netCDF4 as nc
import numpy as np

from Code.Project_Paths import Get_Auxiliary_Data_Path, Get_Model_Path, Get_Satellite_Result_Root
from Code.Satellite_Data_Loaders import Find_First_Matching_File, Load_FY4_AGRI_BT
from Code.Satellite_Retrieval import (
    Retrieval_Region,
    Retrieval_Spec,
    Reorder_ERA_For_FY,
    Run_Satellite_Retrieval,
)

FY_ROOT = os.getenv("GLOBAL_CPP_FY_ROOT", "/home/Data_Pool/data/FY")
AGRI_DIR = os.getenv("GLOBAL_CPP_FY4A_AGRI_ROOT", f"{FY_ROOT}/FY4A/AGRI/4KM")
AGRI_FUNC_LAT_INDEX = np.load(Get_Auxiliary_Data_Path("LAT_4DEG_INDEX_104.npy")).astype(int)
AGRI_FUNC_LON_INDEX = np.load(Get_Auxiliary_Data_Path("LON_4DEG_INDEX_104.npy")).astype(int)

# 【V1.1】延迟加载：SAZ 数组只在首次调用 Load_BT 时从 NC 读取
_AGRI_SAZ = None
SAZ_PATH = Get_Auxiliary_Data_Path("FY4A_AGRI_SAZ_SOZ_SAA_SOA_201908080600_4km.nc")

def _Get_AGRI_SAZ():
    """延迟加载 SAZ 数据，避免在模块导入时打开 NC 文件。"""
    global _AGRI_SAZ
    if _AGRI_SAZ is None:
        with nc.Dataset(SAZ_PATH) as Dataset:
            _AGRI_SAZ = Dataset["SAZ"][:].data
    return _AGRI_SAZ

AGRI_DATA_MINMAX = np.array(
    [[280, 280, 340, 340, 340, 300, 90, 180, 360, 360], [170, 170, 170, 170, 170, 170, 0, 0, 0, 0]]
)
ERA5_LCCS_MINMAX = np.array(
    [[145, 130, 140, 140, 260, 285, 315, 325, 345, 18], [-10, -10, -10, -5, 200, 220, 230, 220, 220, 0]]
)
XMAXMIN = np.hstack([AGRI_DATA_MINMAX[:, [1, 2, 3, 4, 5, 6]], ERA5_LCCS_MINMAX])
Result_Root = Get_Satellite_Result_Root("FY", "FY4A")

Spec = Retrieval_Spec(
    Name="FY4A",
    CPP_Root=str(Result_Root / "CPP"),
    CLP_P_Root=str(Result_Root / "CLP_P"),
    Output_Prefix="FY4A_AGRI",
    Model_Paths=(
        str(Get_Model_Path("CLP", "FY4A_CLP-IR_best.pth")),
        str(Get_Model_Path("CTH", "FY4A_CTH_best.pth")),
        str(Get_Model_Path("CER", "FY4A_CER_best.pth")),
        str(Get_Model_Path("COT", "FY4A_COT_best.pth")),
    ),
    XMAXMIN=XMAXMIN,
    Region=Retrieval_Region(Border_N=81, Border_S=-81, Border_E=186, Border_W=24),
    Overwrite_Existing=False,
    ERA_Transform=Reorder_ERA_For_FY,
)


def Load_BT(Run_Time):
    """定位并读取 FY4A 当前时次的亮温输入。"""
    Pattern = (
        f"{AGRI_DIR}/{Run_Time.year}/{Run_Time.strftime('%Y%m%d')}"
        f"/*{Run_Time.strftime('%Y%m%d%H%M')}*"
    )
    AGRI_Path = Find_First_Matching_File(Pattern)
    return Load_FY4_AGRI_BT(
        AGRI_Path,
        AGRI_FUNC_LAT_INDEX,
        AGRI_FUNC_LON_INDEX,
        _Get_AGRI_SAZ(),  # 【V1.1】只在函数调用时才读取 NC
        Channel_Numbers=(10, 11, 12, 13, 14),
    )


def Retrieval_FY4A_105E_CPP(Run_Time):
    """执行 FY4A 105E CPP 反演。"""
    Run_Satellite_Retrieval(Run_Time, Spec, Load_BT)


if __name__ == "__main__":
    Run_Time = datetime(2020, 11, 25, 0)
    Retrieval_FY4A_105E_CPP(Run_Time)
