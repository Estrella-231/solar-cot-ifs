"""
Description: 加载模型并执行单景云参数反演及结果写出。
Author: Yangzx
LastEditTime: 2026-07-17
Version:
    V1.1_122+local.1
Change:
    1. V1.0: 提供模型加载、云参数反演推理和 NC 结果写出功能。
    2. V1.1: 将 CUDA 初始化从模块级改为延迟加载（Lazy Loading），
             避免 CUDA 初始化导致 HDF5 C 库残留内部对象。
    3. local.1: 使用 inference_mode、修复尾批次 squeeze 风险、避免 Dataset
                原地修改输入；输出改为结构校验并写入结构化日志。
    4. local.1: 补充归一化、tile 数据集和模型输入输出契约注释；算法不变。
"""

import os
import gc
import logging
import time

Thread_Env_Defaults = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

for Env_Name, Env_Value in Thread_Env_Defaults.items():
    os.environ.setdefault(Env_Name, Env_Value)

import numpy as np
import netCDF4 as nc

import torch
import torch.utils.data as Data

import sys
from Project_Paths import Project_Root
sys.path.append(str(Project_Root / 'Model' / 'SmaATUnet'))
from SmaAtUNet import SmaAt_UNet
from Data_Preprocess_FD import Postprocessing, Postprocessing_CLP
from Quality_Control import classify_cloud_phase
from Output_Policy import normalize_output_policy
from Structured_Logging import get_logger, log_event

LOGGER = get_logger("global_cpp.inference")


# 【V1.1】延迟 CUDA 初始化：不再在模块级别调用 torch.cuda.is_available()
DEFAULT_CUDA_INDEX = 5
_Default_Device = None  # 首次调用 _Get_Device() 时初始化

Torch_Num_Threads = int(os.getenv("GLOBAL_CPP_TORCH_THREADS", "1"))
DataLoader_Num_Workers = int(os.getenv("GLOBAL_CPP_DATALOADER_WORKERS", "0"))
Inference_Batch_Size = int(os.getenv("GLOBAL_CPP_BATCH_SIZE", "3"))
if Inference_Batch_Size < 1:
    raise ValueError("GLOBAL_CPP_BATCH_SIZE 必须大于等于1")
torch.set_num_threads(Torch_Num_Threads)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _Get_Device():
    """按环境配置延迟解析并缓存 PyTorch 推理设备。

    优先使用 ``GLOBAL_CPP_DEVICE`` 的完整设备字符串，其次读取
    ``GLOBAL_CPP_CUDA_INDEX``，再回退模块默认 GPU 5。请求索引越界时使用
    ``cuda:0``，无 CUDA 时使用 CPU。结果只解析一次，避免导入模块时初始化 CUDA
    和在同一进程内重复创建设备上下文。
    """
    global _Default_Device
    if _Default_Device is not None:
        return _Default_Device

    Requested_Device = os.getenv("GLOBAL_CPP_DEVICE")
    if Requested_Device:
        _Default_Device = torch.device(Requested_Device)
        return _Default_Device

    Requested_CUDA_Index = os.getenv("GLOBAL_CPP_CUDA_INDEX")
    if torch.cuda.is_available():
        if Requested_CUDA_Index is not None:
            CUDA_Index = int(Requested_CUDA_Index)
        else:
            CUDA_Index = DEFAULT_CUDA_INDEX

        if 0 <= CUDA_Index < torch.cuda.device_count():
            _Default_Device = torch.device(f"cuda:{CUDA_Index}")
            return _Default_Device

        _Default_Device = torch.device("cuda:0")
        return _Default_Device

    _Default_Device = torch.device("cpu")
    return _Default_Device


Cloud_Vmax_CTH = 17.5
Cloud_Vmin_CTH = 0

Cloud_Vmax_CER = 60
Cloud_Vmin_CER = 4

Cloud_Vmax_COT = 100
Cloud_Vmin_COT = 0


def Validate_CPP_File(CPP_Path):
    """轻量验证单星 CPP NetCDF 的结构契约。

    检查 LAT/LON/CLP/CTH/CER/COT 是否存在、经纬度长度是否均为 4051，以及四个
    产品是否为 4051×4051。函数只读取元数据和变量 shape，不扫描整幅数值；通过
    返回 True，结构不符时抛出异常供输出策略决定重算。
    """
    Required_Variables = {"LAT", "LON", "CLP", "CTH", "CER", "COT", "QA"}
    with nc.Dataset(CPP_Path, "r") as Dataset:
        Missing = Required_Variables - set(Dataset.variables)
        if Missing:
            raise ValueError(f"缺少变量: {sorted(Missing)}")
        if len(Dataset.dimensions["LAT"]) != 4051 or len(Dataset.dimensions["LON"]) != 4051:
            raise ValueError("LAT/LON 维度不是 4051×4051")
        if "OBS_INVALID" in Dataset.variables:
            raise ValueError("单星产品不应保存 OBS_INVALID；详细观测异常应写入CSV日志")
        for Name in ("CLP", "CTH", "CER", "COT", "QA"):
            if Dataset.variables[Name].shape != (4051, 4051):
                raise ValueError(f"{Name} 形状异常: {Dataset.variables[Name].shape}")
    return True


def Validate_CLP_P_File(CLP_P_Path):
    """验证单星三分类分数 NPZ 的键名、形状和基础数值契约。

    文件名 ``CLP_P`` 为历史兼容名称，数组实际保存模型 ReLU 后的三通道非负
    分类分数，允许大于 1。视场外位置可以是 NaN；视场内的严格质量判断由全球
    拼接模块结合 CPP 的 CLP 有效掩膜执行。
    """
    with np.load(CLP_P_Path) as Data:
        if "arr_0" not in Data:
            raise ValueError(f"CLP_P 缺少 arr_0: {CLP_P_Path}")
        if Data["arr_0"].shape != (3, 4051, 4051):
            raise ValueError(
                f"CLP_P 形状异常: {Data['arr_0'].shape}，预期 (3,4051,4051)"
            )
    return True


def Check_Output_Pair(CPP_Path, CLP_P_Path, Policy="skip"):
    """根据输出策略判断 CPP/CLP_P 是否可以跳过。

    该函数只读文件系统。结果缺失或损坏时返回 ``False``，由上层重新计算；
    不删除旧文件，便于人工排查和后续恢复。
    """
    Policy = normalize_output_policy(Policy)
    if Policy == "overwrite":
        return False
    if not os.path.isfile(CPP_Path) or not os.path.isfile(CLP_P_Path):
        return False
    try:
        Validate_CPP_File(CPP_Path)
        Validate_CLP_P_File(CLP_P_Path)
        return True
    except Exception as Exc:
        log_event(
            LOGGER, "existing_output_invalid", "已有单星结果校验失败，将重新计算",
            level=logging.WARNING, cpp_path=CPP_Path, clp_probability_path=CLP_P_Path,
            policy=Policy, error=repr(Exc),
        )
        return False


def Check_File(CPP_Path):
    """判断一个已有 CPP 文件是否需要重新反演。

    文件不存在、大小低于 ``GLOBAL_CPP_MIN_CPP_MB`` 或结构校验失败时返回 True；
    文件完整时记录日志并返回 False。函数不主动删除无效文件，后续原子写出会在
    新结果校验通过后替换它，避免重算失败时丢失排查证据。
    """
    if not os.path.exists(CPP_Path):
        return True

    Size_MB = os.path.getsize(CPP_Path) / 1024 / 1024
    Min_Size_MB = float(os.getenv("GLOBAL_CPP_MIN_CPP_MB", "1"))
    try:
        if Size_MB < Min_Size_MB:
            raise ValueError(f"文件仅 {Size_MB:.2f} MB，小于阈值 {Min_Size_MB:.2f} MB")
        Validate_CPP_File(CPP_Path)
    except Exception as Exc:
        log_event(
            LOGGER, "existing_cpp_invalid", "已有单星 CPP 无效，将删除并重算",
            level=logging.WARNING, cpp_path=CPP_Path, size_mb=round(Size_MB, 3), error=repr(Exc),
        )
        return True

    log_event(
        LOGGER, "existing_cpp_valid", "已有单星 CPP 通过校验",
        cpp_path=CPP_Path, size_mb=round(Size_MB, 3),
    )
    return False


class My_Dataset(torch.utils.data.Dataset):
    """按 tile 惰性读取并执行 16 通道 min-max 归一化。

    ``Feed_X`` 应提供 ``[N,16,128,128]`` 索引接口；``XMAXMIN`` 为 ``[2,16]``，
    第 0 行是最大值、第 1 行是最小值。每次取样都会复制 tile，再把 NaN/Inf
    置零，因此不会像旧实现那样原地改写共享输入并导致重复读取数值漂移。
    """
    def __init__(self, Feed_X, XMAXMIN):
        """保存惰性 tile 源并预计算可广播的偏移量与尺度。"""
        self.Imgs = Feed_X
        self.Offset = np.asarray(XMAXMIN[1], dtype=np.float32)[:, None, None]
        self.Scale = np.asarray(XMAXMIN[0] - XMAXMIN[1], dtype=np.float32)[:, None, None]
        if self.Offset.shape[0] != 16 or self.Scale.shape[0] != 16:
            raise ValueError(f"XMAXMIN 应包含 16 个通道，实际 {self.Offset.shape[0]}")
        if np.any(self.Scale == 0):
            raise ValueError("XMAXMIN 存在最大值等于最小值的通道")
        self.Len = len(Feed_X)

    def __len__(self):
        """返回待推理 tile 数。"""
        return self.Len

    def __getitem__(self, Index):
        """返回一个归一化后的 float32 ``[16,128,128]`` tile。"""
        # 必须复制：旧实现会原地归一化 Feed_X，重复读取同一 tile 时数值会再次变化。
        Img_X = np.asarray(self.Imgs[Index], dtype=np.float32).copy()
        np.nan_to_num(Img_X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return (Img_X - self.Offset) / self.Scale


def Retrieval(X_Test, Model_Paths, XMAXMIN):
    """加载四个模型并执行推理，重建单星完整网格产品。

    Parameters
    ----------
    X_Test:
        可索引的 tile 集合，形状契约为 ``[N,16,128,128]``。
    Model_Paths:
        按 CLP、CTH、CER、COT 顺序排列的四个模型权重路径。
    XMAXMIN:
        ``[2,16]`` 的输入归一化上下界，第 0 行为最大值、第 1 行为最小值。

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        CLP 三通道非负分类分数 ``[3,4051,4051]``，以及 CTH、CER、COT 三个
        ``[4051,4051]`` 连续产品。CLP 输出不是 softmax 概率，允许大于 1。

    Notes
    -----
    四个模型在一个 worker 内顺序执行同一批 tile。连续产品按各自训练范围
    反归一化并裁剪，最终卫星视场掩膜在 :func:`Write_To_NC_Using_NC` 中应用。
    """
    Tile_Shape = getattr(X_Test, "shape", None)
    if Tile_Shape is None or len(Tile_Shape) != 4 or tuple(Tile_Shape[1:]) != (16, 128, 128):
        raise ValueError(f"模型 tile 输入应为 [N,16,128,128]，实际 {Tile_Shape}")
    if len(Model_Paths) != 4:
        raise ValueError(f"应提供 CLP/CTH/CER/COT 四个模型，实际 {len(Model_Paths)}")

    Device = _Get_Device()  # V1.1：首次调用时才初始化 CUDA。
    Started = time.perf_counter()
    log_event(
        LOGGER, "inference_started", "开始加载模型并执行推理",
        device=str(Device), tile_count=len(X_Test), batch_size=Inference_Batch_Size,
        model_paths=Model_Paths,
    )

    Nets = [
        SmaAt_UNet(n_channels=16, n_classes=3).to(Device, dtype=torch.float32),
        SmaAt_UNet(n_channels=16, n_classes=1).to(Device, dtype=torch.float32),
        SmaAt_UNet(n_channels=16, n_classes=1).to(Device, dtype=torch.float32),
        SmaAt_UNet(n_channels=16, n_classes=1).to(Device, dtype=torch.float32),
    ]
    Net_CLP = Net_CTH = Net_CER = Net_COT = None
    Test_Dataset = None
    Test_Loader = None
    try:
        for Net, Model_Path in zip(Nets, Model_Paths):
            Checkpoint = torch.load(Model_Path, map_location=Device)
            if "model_state_dict" not in Checkpoint:
                raise KeyError(f"模型缺少 model_state_dict: {Model_Path}")
            Net.load_state_dict(Checkpoint["model_state_dict"])
            Net.eval()
            del Checkpoint

        Test_Dataset = My_Dataset(X_Test, XMAXMIN)
        Test_Loader = Data.DataLoader(
            dataset=Test_Dataset,
            batch_size=Inference_Batch_Size,
            shuffle=False,
            drop_last=False,
            num_workers=DataLoader_Num_Workers,
        )

        Test_NP_CLP, Test_NP_CTH, Test_NP_CER, Test_NP_COT = [], [], [], []
        Net_CLP, Net_CTH, Net_CER, Net_COT = Nets

        # inference_mode 比 no_grad 更彻底地关闭 autograd 元数据，降低显存和 CPU 开销。
        with torch.inference_mode():
            for XX in Test_Loader:
                Inputs_Test = XX.to(device=Device, dtype=torch.float32, non_blocking=True)

                Predict_CLP = Net_CLP(Inputs_Test).cpu().numpy()
                Test_NP_CLP.extend(Predict_CLP)  # 保留 batch 维，避免尾批次为 1 时 squeeze 错位。

                Predict_CTH = Net_CTH(Inputs_Test).cpu().numpy()[:, 0] * (
                    Cloud_Vmax_CTH - Cloud_Vmin_CTH
                ) + Cloud_Vmin_CTH
                Test_NP_CTH.extend(np.clip(Predict_CTH, Cloud_Vmin_CTH, Cloud_Vmax_CTH))

                Predict_CER = Net_CER(Inputs_Test).cpu().numpy()[:, 0] * (
                    Cloud_Vmax_CER - Cloud_Vmin_CER
                ) + Cloud_Vmin_CER
                Test_NP_CER.extend(np.clip(Predict_CER, Cloud_Vmin_CER, Cloud_Vmax_CER))

                Predict_COT = Net_COT(Inputs_Test).cpu().numpy()[:, 0] * (
                    Cloud_Vmax_COT - Cloud_Vmin_COT
                ) + Cloud_Vmin_COT
                Test_NP_COT.extend(np.clip(Predict_COT, Cloud_Vmin_COT, Cloud_Vmax_COT))

                del Inputs_Test

        Cloud_Para_CLP = Postprocessing_CLP(np.asarray(Test_NP_CLP, dtype=np.float32))
        Cloud_Para_CTH = Postprocessing(np.asarray(Test_NP_CTH, dtype=np.float32))
        Cloud_Para_CER = Postprocessing(np.asarray(Test_NP_CER, dtype=np.float32))
        Cloud_Para_COT = Postprocessing(np.asarray(Test_NP_COT, dtype=np.float32))

        log_event(
            LOGGER, "inference_finished", "模型推理和 tile 重建完成",
            device=str(Device), tile_count=len(X_Test),
            elapsed_seconds=round(time.perf_counter() - Started, 3),
        )
        return Cloud_Para_CLP, Cloud_Para_CTH, Cloud_Para_CER, Cloud_Para_COT
    except Exception as Exc:
        log_event(
            LOGGER, "inference_failed", "模型加载或推理失败",
            level=logging.ERROR, exc_info=True, device=str(Device), error=repr(Exc),
            elapsed_seconds=round(time.perf_counter() - Started, 3),
        )
        raise
    finally:
        Net_CLP = Net_CTH = Net_CER = Net_COT = None
        Test_Dataset = None
        Test_Loader = None
        for Index in range(len(Nets)):
            Nets[Index] = None
        Nets.clear()
        gc.collect()
        if Device.type == "cuda":
            torch.cuda.empty_cache()


def Write_To_NC_Using_NC(CPP, Observation_Invalid, QA, CPP_Path, CLP_P_Path, Border_N, Border_S, Border_E, Border_W):
    """校验、掩膜并原子写出单星 CPP NetCDF 与 CLP_P NPZ。

    ``CPP`` 顺序必须为 CLP 三通道分类分数、CTH、CER、COT。观测无效像元已在
    上层按上一小时产品完成回填；本函数不再以单个亮温通道二次清空这些结果。
    原始观测异常只参与内存质量控制，不写入NetCDF；``QA``标记实际完成的
    上一时次回填。更详细的异常原因和统计量由结构化日志或CSV保存。

    CPP 与 CLP_P 分别写到同目录临时文件，完成结构回读校验后使用 ``os.replace``
    原子发布。每类文件最多重试三次；任一文件最终失败都会抛出异常。
    """
    Expected_Shapes = ((3, 4051, 4051), (4051, 4051), (4051, 4051), (4051, 4051))
    if len(CPP) != 4:
        raise ValueError(f"CPP 应包含四项输出，实际 {len(CPP)}")
    for Index, (Value, Expected) in enumerate(zip(CPP, Expected_Shapes)):
        if np.asarray(Value).shape != Expected:
            raise ValueError(f"CPP[{Index}] 形状应为 {Expected}，实际 {np.asarray(Value).shape}")
    for Name, Value in (("Observation_Invalid", Observation_Invalid), ("QA", QA)):
        if np.asarray(Value).shape != (4051, 4051):
            raise ValueError(f"{Name} 应为 4051×4051，实际 {np.asarray(Value).shape}")

    CLP_Scores = np.asarray(CPP[0], dtype=np.float32).copy()
    CLP = classify_cloud_phase(CLP_Scores)
    CTH = CPP[1].squeeze().astype(np.float32)
    CER = CPP[2].squeeze().astype(np.float32)
    COT = CPP[3].squeeze().astype(np.float32)
    CTH[np.isfinite(CTH)] = np.clip(CTH[np.isfinite(CTH)], 0.01, 17.5)
    CER[np.isfinite(CER)] = np.clip(CER[np.isfinite(CER)], 4, 60)
    COT[np.isfinite(COT)] = np.clip(COT[np.isfinite(COT)], 0.01, 100)
    CTH[CLP == 0] = 0
    CER[CLP == 0] = 0
    COT[CLP == 0] = 0
    Observation_Invalid = np.asarray(Observation_Invalid, dtype=np.int8)
    QA = np.asarray(QA, dtype=np.int8)
    LON = np.linspace(Border_W, Border_E, 4051)
    LAT = np.linspace(Border_N, Border_S, 4051)

    Retry_Delay = float(os.getenv("GLOBAL_CPP_WRITE_RETRY_DELAY", "1"))
    Temp_CPP_Path = f"{CPP_Path}.tmp.{os.getpid()}"
    Temp_CLP_P_Path = f"{CLP_P_Path}.tmp.{os.getpid()}.npz"

    # 先写同目录临时文件，通过校验后原子替换，避免失败时破坏已有结果。
    for attempt in range(1, 4):
        try:
            if os.path.exists(Temp_CPP_Path):
                os.remove(Temp_CPP_Path)
            with nc.Dataset(Temp_CPP_Path, 'w', format='NETCDF4') as DS:
                DS.createDimension('LAT', 4051)
                DS.createDimension('LON', 4051)
                DS.createVariable('LAT', np.float32, dimensions=('LAT',))
                DS.createVariable('LON', np.float32, dimensions=('LON',))
                DS.createVariable('CER', np.float32, ('LAT', 'LON'), zlib=True, complevel=3)
                DS.createVariable('COT', np.float32, ('LAT', 'LON'), zlib=True, complevel=3)
                DS.createVariable('CTH', np.float32, ('LAT', 'LON'), zlib=True, complevel=3)
                DS.createVariable('CLP', np.float32, ('LAT', 'LON'), zlib=True, complevel=3)
                DS.createVariable('QA', np.int8, ('LAT', 'LON'), zlib=True, complevel=3)
                DS.variables['LON'][:] = LON
                DS.variables['LAT'][:] = LAT
                DS.variables['CLP'][:] = CLP
                DS.variables['CTH'][:] = CTH
                DS.variables['CER'][:] = CER
                DS.variables['COT'][:] = COT
                DS.variables['QA'][:] = QA
                DS.variables['LON'].setncatts({'Long_Name': 'Longitude'})
                DS.variables['LAT'].setncatts({'Long_Name': 'Latitude'})
                DS.variables['CLP'].setncatts({
                    'Long_Name': 'Cloud Phase',
                    'Range': '0: Clear; 1: Water; 2: Ice',
                    'Unit': 'None'})
                DS.variables['CTH'].setncatts({
                    'Long_Name': 'Cloud Top Height',
                    'Range': '0 - 17.5',
                    'Unit': 'km'})
                DS.variables['CER'].setncatts({
                    'Long_Name': 'Cloud Effective Radius',
                    'Range': '0 - 60',
                    'Unit': 'μm'})
                DS.variables['COT'].setncatts({
                    'Long_Name': 'Cloud Optical Thickness',
                    'Range': '0 - 100',
                    'Unit': 'None'})
                DS.variables['QA'].setncatts({'Long_Name': 'Previous-hour replacement flag', 'Range': '0: native current result; 1: filled from previous hour', 'Unit': 'None'})
            Validate_CPP_File(Temp_CPP_Path)
            os.replace(Temp_CPP_Path, CPP_Path)
            file_size = os.path.getsize(CPP_Path) / (1024 * 1024)
            log_event(
                LOGGER, "satellite_nc_written", "单星 NetCDF 写出并校验成功",
                cpp_path=CPP_Path, size_mb=round(file_size, 3), attempt=attempt,
            )
            break  # 成功，跳出循环
        except Exception as e:
            log_event(
                LOGGER, "satellite_nc_write_retry", "单星 NetCDF 写出失败",
                level=logging.WARNING if attempt < 3 else logging.ERROR,
                cpp_path=CPP_Path, attempt=attempt, max_attempts=3, error=repr(e),
            )
            if os.path.exists(Temp_CPP_Path):
                os.remove(Temp_CPP_Path)
            if attempt == 3:
                raise RuntimeError(f'NC write failed after 3 attempts: {e}')
            time.sleep(Retry_Delay)

    # CLP 三通道分类分数单独写 NPZ。历史目录名 CLP_P 保持不变，但这些值
    # 没有经过 softmax，可以大于 1；只验证键名、形状和文件可重新打开。
    for attempt in range(1, 4):
        try:
            if os.path.exists(Temp_CLP_P_Path):
                os.remove(Temp_CLP_P_Path)
            np.savez_compressed(Temp_CLP_P_Path, CLP_Scores)
            with np.load(Temp_CLP_P_Path) as NPZ_Data:
                if "arr_0" not in NPZ_Data or NPZ_Data["arr_0"].shape != (3, 4051, 4051):
                    raise ValueError("NPZ 缺少 arr_0 或概率形状异常")
            os.replace(Temp_CLP_P_Path, CLP_P_Path)
            file_size = os.path.getsize(CLP_P_Path) / (1024 * 1024)
            log_event(
                LOGGER, "satellite_npz_written", "单星 CLP 分类分数 NPZ 写出并校验成功",
                clp_probability_path=CLP_P_Path, size_mb=round(file_size, 3), attempt=attempt,
            )
            break
        except Exception as e:
            log_event(
                LOGGER, "satellite_npz_write_retry", "单星 CLP 分类分数 NPZ 写出失败",
                level=logging.WARNING if attempt < 3 else logging.ERROR,
                clp_probability_path=CLP_P_Path,
                attempt=attempt, max_attempts=3, error=repr(e),
            )
            if os.path.exists(Temp_CLP_P_Path):
                os.remove(Temp_CLP_P_Path)
            if attempt == 3:
                raise RuntimeError(f'NPZ write failed after 3 attempts: {e}')
            time.sleep(Retry_Delay)
