"""统一网格切片与反演结果重建。

模块职责：
    将 ``[C, 4051, 4051]`` 单星输入惰性切分为 63×63 个 128×128 tile；
    将模型输出按 64 像元重叠区平滑融合，裁剪回 4051×4051 业务网格。
输入/输出约定：
    输入统一使用 ``[channel, latitude, longitude]``；单参数输出为二维数组，
    云相分类分数 CLP_P 输出为 ``[3, latitude, longitude]``。内部计算使用 float32。
关键约束：
    tile 步长固定为 64，尺寸固定为 128，因此每个方向需要 63 个 tile。
    4051 不能被步长整除，最南侧和最东侧 tile 在访问时临时补零，最终再裁剪。
版本历史：
    V1.1_122          原始切片与融合实现。
    V1.1_122+local.1  改为惰性取块，避免提前物化数 GB tile；移除固定卫星掩膜；
                         细化数据形状、边界补零和融合规则注释。
"""

import numpy as np


class Tile_Collection:
    """4051 原始网格上的惰性 128×128 tile 集合。

    对象只引用输入数组，不创建完整的 4096×4096 补边副本。DataLoader 按线性
    索引取样时，先换算成 63×63 网格中的行列位置；内部 tile 尽量返回原数组
    视图，只有越过南/东边界的 tile 才分配一个临时 float32 零填充数组。

    Parameters
    ----------
    Data : numpy.ndarray
        形状为 ``[C, H, W]`` 的通道优先数组。业务调用中 H=W=4051；类本身
        只校验维数，严格网格尺寸由 :func:`Cut` 负责校验。

    Notes
    -----
    ``shape`` 模拟普通 ndarray 的 ``[N,C,H,W]`` 属性，供后续 Dataset 使用；
    N 固定为 3969。返回的边界 tile 中，超出业务网格的部分为 0。
    """

    Grid_Count = 63
    Tile_Size = 128
    Step = 64

    def __init__(self, Data):
        """保存输入引用并建立虚拟 tile 集合的形状元数据。"""
        if Data.ndim != 3:
            raise ValueError(f"Tile_Collection 输入应为 [C,H,W]，实际 {Data.shape}")
        self.Data = Data
        self.shape = (
            self.Grid_Count * self.Grid_Count,
            Data.shape[0], self.Tile_Size, self.Tile_Size,
        )

    def __len__(self):
        """返回 tile 总数 3969（63×63）。"""
        return self.shape[0]

    def __getitem__(self, Index):
        """按行优先线性索引返回一个 ``[C,128,128]`` tile。

        支持 Python 负索引；越界时抛出 ``IndexError``。非边界位置通常返回
        原始数组视图，调用方如需原地修改必须先复制，避免污染后续 tile。
        """
        if Index < 0:
            Index += len(self)
        if Index < 0 or Index >= len(self):
            raise IndexError(Index)
        Row, Col = divmod(Index, self.Grid_Count)
        Row_Start = Row * self.Step
        Col_Start = Col * self.Step
        Row_End = min(Row_Start + self.Tile_Size, self.Data.shape[1])
        Col_End = min(Col_Start + self.Tile_Size, self.Data.shape[2])
        Source = self.Data[:, Row_Start:Row_End, Col_Start:Col_End]
        if Source.shape[1:] == (self.Tile_Size, self.Tile_Size):
            return Source

        # 只有最南/最东边界会进入此分支；临时分配单个 tile，而非完整补边网格。
        Tile = np.zeros((self.Data.shape[0], self.Tile_Size, self.Tile_Size), dtype=np.float32)
        Tile[:, :Source.shape[1], :Source.shape[2]] = Source
        return Tile


def Cut(Data):
    """校验统一网格并返回惰性 tile 集合。

    Parameters
    ----------
    Data : array-like
        ``[C,4051,4051]`` 输入。若可安全复用则不复制；非 float32 输入会转换。

    Returns
    -------
    Tile_Collection
        可按索引读取的 3969 个 ``[C,128,128]`` tile。
    """
    Data_Array = np.asarray(Data)
    if Data_Array.ndim != 3 or Data_Array.shape[1:] != (4051, 4051):
        raise ValueError(f"Cut 输入应为 [C,4051,4051]，实际 {Data_Array.shape}")
    return Tile_Collection(Data_Array.astype(np.float32, copy=False))


def Cal_Weight(D, K):
    """生成长度为 ``D`` 的 S 形重叠区融合权重。

    ``K`` 控制过渡陡峭程度；权重从接近 0 单调增至接近 1，使左/上图像的
    权重逐渐减小、右/下图像的权重逐渐增大。返回 float32 一维数组。
    """
    X = np.arange(-D / 2, D / 2, dtype=np.float32)
    return (1 / (1 + np.exp(-K * X / D))).astype(np.float32)


def Img_Fusion(Img1, Img2, Overlap, Left_Right=True):
    """融合两个相邻二维 tile 或已融合条带。

    Parameters
    ----------
    Img1, Img2 : numpy.ndarray
        待拼接的二维数组。横向拼接时行数必须一致；纵向拼接时列数必须一致。
    Overlap : int
        两数组的重叠像元数，当前生产流程固定为 64。
    Left_Right : bool
        True 表示 Img2 位于 Img1 右侧，False 表示 Img2 位于 Img1 下方。

    Returns
    -------
    numpy.ndarray
        使用 sigmoid 权重平滑过渡后的 float32 数组；重叠区只保留一份。
    """
    Weight = Cal_Weight(Overlap, 5)

    if Left_Right:
        Row, Col = Img1.shape
        Img_New = np.zeros((Row, Col + 64), dtype=np.float32)
        Img_New[:, :Col] = Img1
        Weight_Expand = np.tile(Weight, (Row, 1))
        Img_New[:, Col - Overlap:Col] = (
            (1 - Weight_Expand) * Img1[:, Col - Overlap:Col]
            + Weight_Expand * Img2[:, :Overlap]
        )
        Img_New[:, Col:] = Img2[:, Overlap:]
    else:
        Row, Col = Img1.shape
        Img_New = np.zeros((Row + 64, Col), dtype=np.float32)
        Img_New[:Row, :] = Img1
        Weight = np.reshape(Weight, (Overlap, 1))
        Weight_Expand = np.tile(Weight, (1, Col))
        Img_New[Row - Overlap:Row, :] = (
            (1 - Weight_Expand) * Img1[Row - Overlap:Row, :]
            + Weight_Expand * Img2[:Overlap, :]
        )
        Img_New[Row:, :] = Img2[Overlap:, :]
    return Img_New


def Postprocessing(CP):
    """把单通道 tile 按 sigmoid 权重重建为 4051×4051。"""
    Cloud_Parameter = CP
    Row_List = []
    for M in range(63):
        Row_Merge = Cloud_Parameter[63 * M]
        for N in range(M * 63 + 1, 63 + M * 63):
            Tile_Data = Cloud_Parameter[N]
            Row_Merge = Img_Fusion(Row_Merge, Tile_Data, 64, Left_Right=True)
        Row_List.append(Row_Merge)

    Merged_Data = Row_List[0]
    for C in range(1, 63):
        Merged_Data = Img_Fusion(Merged_Data, Row_List[C], 64, Left_Right=False)

    Output_Data = Merged_Data[0:4051, 0:4051].astype(np.float32, copy=False)
    return Output_Data


def Postprocessing_CLP(CP):
    """把三通道 CLP 分类分数 tile 重建为 ``[3,4051,4051]``。

    输入来自模型 ReLU 输出，不执行 softmax；重建结果是可大于 1 的非负分数，
    后续仅以通道 argmax 判定 0/1/2 云相类别。
    """
    Cloud_Probability = CP
    Channel_List = []
    for T in range(len(Cloud_Probability[0])):
        Channel_Data = Cloud_Probability[:, T]
        Row_List = []
        for M in range(63):
            Row_Merge = Channel_Data[63 * M]
            for N in range(M * 63 + 1, 63 + M * 63):
                Tile_Data = Channel_Data[N]
                Row_Merge = Img_Fusion(Row_Merge, Tile_Data, 64, Left_Right=True)
            Row_List.append(Row_Merge)

        Merged_Data = Row_List[0]
        for C in range(1, 63):
            Merged_Data = Img_Fusion(Merged_Data, Row_List[C], 64, Left_Right=False)

        Channel_List.append(Merged_Data)

    Output_Data = np.asarray(Channel_List, dtype=np.float32)[:, 0:4051, 0:4051]
    return Output_Data
