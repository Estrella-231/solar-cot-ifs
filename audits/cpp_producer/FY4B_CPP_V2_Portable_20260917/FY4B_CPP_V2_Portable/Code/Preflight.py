"""Global CPP 运行前静态检查。

模块职责：在真正读取大体量卫星数据或创建工作进程之前，检查 Python 依赖、
四类模型权重、公共辅助数据和可选数据根目录，尽早给出可操作的问题列表。
该模块只读文件系统，不创建目录、不加载模型，也不会修改输入数据。

版本历史：
    V1.1_122+local.1  新增统一入口所需的运行前检查，并补充检查范围、返回语义
                         和模型路径映射注释。
"""

import importlib.util
import os
from pathlib import Path
from typing import Iterable, List, Sequence

from Project_Paths import Auxiliary_Data_Dir, Get_Model_Path
from Satellite_Timeline import KNOWN_SATELLITES


MODEL_NAMES = {
    "FY4A_105E": ("FY4A_CLP-IR_best.pth", "FY4A_CTH_best.pth", "FY4A_CER_best.pth", "FY4A_COT_best.pth"),
    "FY4B_105E": ("FY4B_CLP-IR_best.pth", "FY4B_CTH_best.pth", "FY4B_CER_best.pth", "FY4B_COT_best.pth"),
    "FY4B_133E": ("FY4B_CLP-IR_best.pth", "FY4B_CTH_best.pth", "FY4B_CER_best.pth", "FY4B_COT_best.pth"),
    "GOES16": ("GOES-16_CLP-IR_best.pth", "GOES-16_CTH_best.pth", "GOES-16_CER_best.pth", "GOES-16_COT_best.pth"),
    "GOES17": ("GOES-17_CLP-IR_best.pth", "GOES-17_CTH_best.pth", "GOES-17_CER_best.pth", "GOES-17_COT_best.pth"),
    "GOES18": ("GOES-18_CLP-IR_best.pth", "GOES-18_CTH_best.pth", "GOES-18_CER_best.pth", "GOES-18_COT_best.pth"),
    "MSG1_IODC": ("MSG1_IODC_CLP-IR_V2_best.pth", "MSG1_IODC_CTH_V2_best.pth", "MSG1_IODC_CER_V2_best.pth", "MSG1_IODC_COT_V2_best.pth"),
    "MSG2_IODC": ("MSG2_IODC_CLP-IR_V2_best.pth", "MSG2_IODC_CTH_V2_best.pth", "MSG2_IODC_CER_V2_best.pth", "MSG2_IODC_COT_V2_best.pth"),
    "MSG3_0DEG": ("MSG3_0DEG_CLP-IR_V2_best.pth", "MSG3_0DEG_CTH_V2_best.pth", "MSG3_0DEG_CER_V2_best.pth", "MSG3_0DEG_COT_V2_best.pth"),
    "MSG4_0DEG": ("MSG4_0DEG_CLP-IR_V2_best.pth", "MSG4_0DEG_CTH_V2_best.pth", "MSG4_0DEG_CER_V2_best.pth", "MSG4_0DEG_COT_V2_best.pth"),
}


def _model_paths(satellite: str) -> Iterable[Path]:
    """按 CLP、CTH、CER、COT 的固定顺序生成某颗卫星的权重路径。

    Parameters
    ----------
    satellite : str
        :mod:`Satellite_Timeline` 中登记的规范卫星名称。

    Yields
    ------
    pathlib.Path
        四个只读模型文件路径。未知名称由字典访问直接抛出 ``KeyError``，调用方
        应先完成名称校验。
    """
    clp, cth, cer, cot = MODEL_NAMES[satellite]
    yield Path(Get_Model_Path("CLP", clp))
    yield Path(Get_Model_Path("CTH", cth))
    yield Path(Get_Model_Path("CER", cer))
    yield Path(Get_Model_Path("COT", cot))


def run_preflight(satellites: Sequence[str], check_data_roots: bool = True) -> List[str]:
    """检查指定卫星运行所需的静态资源。

    Parameters
    ----------
    satellites : Sequence[str]
        待执行的规范卫星名称，可同时包含多颗卫星。
    check_data_roots : bool
        是否检查 FY/GOES/MSG/ERA5 数据根目录。只做部署检查或测试时可关闭。

    Returns
    -------
    list[str]
        所有发现的问题；空列表仅表示静态检查通过，不保证目标时次原始文件齐全。
        函数会尽量收集全部问题，而不是发现第一个问题后立即退出。
    """
    problems: List[str] = []
    required_packages = ("numpy", "pandas", "xarray", "netCDF4", "torch")
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            problems.append(f"缺少 Python 依赖: {package}")
    if any(name.startswith("MSG") for name in satellites) and importlib.util.find_spec("satpy") is None:
        problems.append("MSG/SEVIRI 反演需要 satpy")
    for satellite in satellites:
        if satellite not in KNOWN_SATELLITES:
            problems.append(f"未知卫星: {satellite}")
            continue
        for model_path in _model_paths(satellite):
            if not model_path.is_file():
                problems.append(f"缺少模型: {model_path}")
    if not Auxiliary_Data_Dir.is_dir():
        problems.append(f"缺少辅助数据目录: {Auxiliary_Data_Dir}")
    if check_data_roots:
        roots = {
            "FY": os.getenv("GLOBAL_CPP_FY_ROOT", "/home/Data_Pool/data/FY"),
            "GOES": os.getenv("GLOBAL_CPP_GOES_ROOT", "/home/Data_Pool/data/GOES"),
            "MSG": os.getenv("GLOBAL_CPP_MSG_ROOT", "/home/Data_Pool/data/Meteosat"),
            "ERA5": os.getenv("GLOBAL_CPP_ERA5_ROOT", "/home/Data_Pool/data/ERA5"),
        }
        for label, raw_path in roots.items():
            if not Path(raw_path).exists():
                problems.append(f"{label} 数据根目录不存在: {raw_path}")
    return problems
