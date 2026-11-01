"""CPP/CLP_P 产品级质量控制与上一时刻像元回退。

本模块只处理“产品值是否可信”和“是否需要上一时刻填补”两件事，
不负责卫星覆盖范围判定，也不负责全球经纬度拼接。调用方需要传入同一
覆盖区、同一网格上的当前时刻和上一时刻产品。

质量控制规则
-------------
* CLP 必须是 0、1、2；CTH、CER、COT 必须在产品物理范围内。
* NaN、无穷大、超出范围的值都视为无效。
* 对云像元，CTH、CER、COT 的 0 值视为缺测；晴空像元的 0 值保留，
  因为当前产品约定晴空时这些连续参数为 0。
* 历史文件名沿用 ``CLP_P``，但其中保存的是模型 ReLU 后的三通道分类分数，
  不是和为 1 的概率。分数允许大于 1；只有 NaN/Inf、负值或三个通道全为 0
  才视为无效。CLP_P 异常时，使用上一时刻同一像元的完整三通道分数。
* QA=1 只表示对应位置确实从上一时刻复制成功；两时刻都无有效值时保留
  NaN，QA=0，避免把“未填补”误报为“已填补”。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


PRODUCT_LIMITS = {
    "CLP": (0.0, 2.0),
    "CTH": (0.0, 17.5),
    # 晴空像元的 CER/COT 为 0；有云像元由反演阶段限制到物理范围。
    "CER": (0.0, 60.0),
    "COT": (0.0, 100.0),
}
PRODUCT_NAMES = tuple(PRODUCT_LIMITS)
CLOUD_PRODUCTS = ("CTH", "CER", "COT")
CLOUD_VALID_MINIMUMS = {
    # CTH/COT 的单星写出下限为 0.01，因此云像元必须严格大于 0。
    "CTH": (0.0, False),
    # CER 模型按训练范围反归一化并裁剪到 [4, 60]。
    "CER": (4.0, True),
    "COT": (0.0, False),
}


def _validate_product_shape(values: np.ndarray, name: str, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """将产品转为二维 ``float32`` 数组并检查网格形状。

    参数
    ----
    values:
        输入产品数组，期望为 ``(lat, lon)``。
    name:
        产品名称，仅用于生成可读错误信息。
    expected_shape:
        若提供，则要求数组与该网格形状一致。
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组 (lat, lon)，实际为 {array.shape}")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(f"{name} 网格形状不一致: {array.shape} != {expected_shape}")
    return array


def _invalid_product(
    values: np.ndarray,
    name: str,
    cloud_mask: np.ndarray | None = None,
) -> np.ndarray:
    """返回单个产品的逐像元无效掩膜。

    ``0`` 对 CTH/CER/COT 并不总是错误：晴空像元用 0 表示“该参数不适用”。
    但在 CLP_P 判定为云的像元中，连续参数为 0 通常代表反演缺测，因此额外
    标记为无效，并允许从上一时刻回退。这样既不会把全球大量晴空 0 错误替换，
    也不会遗漏云区的 0 值缺测。
    """
    values = _validate_product_shape(values, name)
    lower, upper = PRODUCT_LIMITS[name]
    invalid = ~np.isfinite(values) | (values < lower) | (values > upper)
    if name == "CLP":
        finite = np.isfinite(values)
        invalid |= finite & (np.abs(values - np.rint(values)) > 1e-4)
    elif name in CLOUD_PRODUCTS and cloud_mask is not None:
        cloud_mask = np.asarray(cloud_mask, dtype=bool)
        if cloud_mask.shape != values.shape:
            raise ValueError(f"{name} 的云掩膜形状不一致: {cloud_mask.shape} != {values.shape}")
        minimum, inclusive = CLOUD_VALID_MINIMUMS[name]
        if inclusive:
            below_cloud_minimum = values < minimum
        else:
            below_cloud_minimum = values <= minimum
        invalid |= cloud_mask & np.isfinite(values) & below_cloud_minimum
    return invalid


def _invalid_probability(probabilities: np.ndarray) -> np.ndarray:
    """返回 CLP_P 三通道分类分数无效的逐像元掩膜。

    ``CLP_P`` 是为兼容 V1 文件结构保留的历史名称。实际数组来自模型最后一层
    ReLU，保存的是非负分类分数而非 softmax 概率，所以数值可以大于 1，也不要求
    三通道之和等于 1。以下情况才判为无效：

    * 数组不是 ``(3, lat, lon)``；
    * 任一通道为 NaN/Inf 或小于 0；
    * 三个通道全部为 0，无法给出有效类别。

    将大于 1 的正常分数误判为坏值会使整片 CLP_P 变成 NaN，而 NumPy 对全 NaN
    直接 ``argmax`` 又会返回类别 0，最终造成全球产品大面积假晴空和连续参数 0。
    """
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"CLP_P 形状必须为 (3, lat, lon)，实际为 {values.shape}")
    invalid_channel = ~np.isfinite(values) | (values < 0.0)
    no_class_score = np.all(values == 0.0, axis=0)
    return np.any(invalid_channel, axis=0) | no_class_score


def classify_cloud_phase(scores: np.ndarray) -> np.ndarray:
    """把三通道 CLP 分类分数安全转换为二维云相类别。

    Parameters
    ----------
    scores:
        ``(3, lat, lon)`` 的非负分类分数。有效像元取最大分数对应类别：0 为
        晴空、1 为液态云、2 为冰云。

    Returns
    -------
    numpy.ndarray
        ``float32`` 二维数组。有效位置为 0/1/2；分数异常的位置保持 NaN。
        该显式有效性判断避免 ``np.argmax`` 把全 NaN 像元静默归为类别 0。
    """
    values = np.asarray(scores, dtype=np.float32)
    invalid = _invalid_probability(values)
    phase = np.full(values.shape[1:], np.nan, dtype=np.float32)
    valid = ~invalid
    phase[valid] = np.argmax(values[:, valid], axis=0).astype(np.float32)
    return phase


def scores_to_probabilities(scores: np.ndarray) -> np.ndarray:
    """将三通道非负分类分数稳定转换为三分类概率。

    CLP_P 的历史模型输出是 ReLU 后的非负分数，内部拼接仍使用分数以保持现有
    argmax 结果；对外提供给 MODIS 等其他传感器融合时，则需要真正和为 1 的概率。
    本函数沿类别维执行数值稳定的 softmax，忽略无效像元并保持其为 NaN。

    Parameters
    ----------
    scores:
        形状为 ``(3, lat, lon)`` 的非负分类分数，可包含视场外 NaN。

    Returns
    -------
    numpy.ndarray
        与输入同形状的 float32 概率。有效像元三个类别之和约等于 1；无效像元
        三个类别均为 NaN。减去类别最大值后再指数运算，避免大分数溢出。
    """
    values = np.asarray(scores, dtype=np.float32)
    invalid = _invalid_probability(values)
    probabilities = np.full(values.shape, np.nan, dtype=np.float32)
    valid = ~invalid
    if not np.any(valid):
        return probabilities
    valid_scores = values[:, valid]
    shifted = valid_scores - np.max(valid_scores, axis=0, keepdims=True)
    exponent = np.exp(shifted, dtype=np.float32)
    denominator = np.sum(exponent, axis=0, keepdims=True)
    stable = np.isfinite(denominator[0]) & (denominator[0] > 0)
    if np.any(stable):
        valid_indices = np.flatnonzero(valid)
        probabilities.reshape(3, -1)[:, valid_indices[stable]] = (
            exponent[:, stable] / denominator[:, stable]
        ).astype(np.float32)
    return probabilities


def _cloud_mask(probabilities: np.ndarray) -> np.ndarray:
    """根据有效 CLP_P 计算云像元掩膜。

    概率最大类别为 0 时视为晴空，最大类别为 1 或 2 时视为云。概率异常
    的位置不参与该判断，避免把结构性 NaN 或坏概率误当作云区。
    """
    phase = classify_cloud_phase(probabilities)
    return np.isfinite(phase) & (phase != 0)


def apply_previous_observation_fallback(
    current_values: Mapping[str, np.ndarray],
    current_clpp: np.ndarray,
    observation_invalid: np.ndarray,
    previous_values: Mapping[str, np.ndarray] | None,
    previous_clpp: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, int]]:
    """按原始观测无效标志使用上一小时完整单星产品回填。

    ``observation_invalid`` 是传感器加载阶段由原始质量标志、填充值、无效定标值
    和视场外几何掩膜共同形成的二维掩膜。它是单星回填的唯一触发条件：模型输出
    数值本身不会在这里触发回填。若上一小时同一像元的 CLP_P、CLP 和云相对应的
    连续产品完整有效，则四项产品一起复制，QA 置 1；否则当前像元保持 NaN。
    未解决数量由调用方写入结构化日志或CSV，不额外保存OBS_INVALID变量。
    """
    missing = [name for name in PRODUCT_NAMES if name not in current_values]
    if missing:
        raise ValueError(f"当前产品缺少变量: {missing}")
    updated = {name: _validate_product_shape(current_values[name], name).copy() for name in PRODUCT_NAMES}
    shape = updated["CLP"].shape
    scores = np.asarray(current_clpp, dtype=np.float32).copy()
    if scores.shape != (3, *shape):
        raise ValueError(f"当前 CLP_P 形状异常: {scores.shape}，预期 {(3, *shape)}")
    invalid = np.asarray(observation_invalid, dtype=bool)
    if invalid.shape != shape:
        raise ValueError(f"观测无效掩膜形状异常: {invalid.shape}，预期 {shape}")

    # 被观测标志拒绝的像元绝不保留本时次模型推理值。
    for values in updated.values():
        values[invalid] = np.nan
    scores[:, invalid] = np.nan
    qa = np.zeros(shape, dtype=bool)
    if previous_values is None or previous_clpp is None or not np.any(invalid):
        return updated, scores, qa, {
            "observation_invalid_pixels": int(invalid.sum()),
            "observation_filled_pixels": 0,
            "observation_unresolved_pixels": int(invalid.sum()),
        }

    # 先用已有的产品规则清洁上一小时，保证链式回填只引用有效或此前已填补的值。
    previous, previous_scores, _, _ = apply_previous_quality_fallback(
        previous_values, previous_clpp, None, None
    )
    previous_phase = classify_cloud_phase(previous_scores)
    source_complete = np.isfinite(previous_phase) & (~_invalid_probability(previous_scores))
    previous_cloud = np.isfinite(previous_phase) & (previous_phase != 0)
    for name in CLOUD_PRODUCTS:
        source_complete &= ~_invalid_product(previous[name], name, previous_cloud)
    source_complete &= ~_invalid_product(previous["CLP"], "CLP")
    replace = invalid & source_complete
    for name in PRODUCT_NAMES:
        updated[name][replace] = previous[name][replace]
    scores[:, replace] = previous_scores[:, replace]
    qa[replace] = True
    return updated, scores, qa, {
        "observation_invalid_pixels": int(invalid.sum()),
        "observation_filled_pixels": int(replace.sum()),
        "observation_unresolved_pixels": int((invalid & ~replace).sum()),
    }


def apply_previous_quality_fallback(
    current_values: Mapping[str, np.ndarray],
    current_clpp: np.ndarray,
    previous_values: Mapping[str, np.ndarray] | None,
    previous_clpp: np.ndarray | None,
    processing_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    """质量控制并按变量/像元使用上一时刻结果替换。

    参数必须是同一覆盖区、同一网格的当前/上一时刻数据。产品回退是“按变量、
    按像元”进行的：例如当前 CTH 缺测而 CER/COT 有效时，只替换 CTH，不覆盖
    CER/COT。只有上一时刻对应变量的对应像元有效且确实完成替换时才置 QA=1；
    当前和上一时刻都无有效像元的位置保持 NaN，不伪造填补标志。

    ``processing_mask``可选，用于把处理限制在卫星真实视场内。掩膜外的NaN是
    圆盘外背景，不是产品缺测，不能从上一时刻填补，也不能因此设置QA。

    返回
    ----
    updated:
        经过质量控制和回退后的四个产品数组。
    updated_clpp:
        经过质量控制和回退后的三通道 CLP_P。
    replacement_mask:
        所有产品的替换掩膜并集；True 对应输出 QA=1。
    stats:
        总体和按产品统计，包括发现的坏像元、成功替换像元和仍未解决像元。
    """
    missing = [name for name in PRODUCT_NAMES if name not in current_values]
    if missing:
        raise ValueError(f"当前产品缺少变量: {missing}")

    updated = {name: _validate_product_shape(current_values[name], name).copy() for name in PRODUCT_NAMES}
    updated_clpp = np.asarray(current_clpp, dtype=np.float32).copy()
    if updated_clpp.ndim != 3 or updated_clpp.shape[0] != 3:
        raise ValueError(f"当前 CLP_P 形状必须为 (3, lat, lon)，实际为 {updated_clpp.shape}")
    expected_shape = updated["CLP"].shape
    if updated_clpp.shape[1:] != expected_shape:
        raise ValueError(f"当前 CLP_P 与产品网格不一致: {updated_clpp.shape[1:]} != {expected_shape}")
    if processing_mask is None:
        allowed = np.ones(expected_shape, dtype=bool)
    else:
        allowed = np.asarray(processing_mask, dtype=bool)
        if allowed.shape != expected_shape:
            raise ValueError(
                f"处理区域掩膜形状不一致: {allowed.shape} != {expected_shape}"
            )
    current_cloud = _cloud_mask(updated_clpp)
    current_invalid = {name: _invalid_product(updated[name], name, current_cloud) for name in PRODUCT_NAMES}
    current_invalid["CLP_P"] = _invalid_probability(updated_clpp)
    for name in current_invalid:
        current_invalid[name] &= allowed

    previous_valid: dict[str, np.ndarray] = {}
    previous_cloud = None
    if previous_values is not None:
        for name in PRODUCT_NAMES:
            if name in previous_values:
                _validate_product_shape(previous_values[name], name, expected_shape)
        if previous_clpp is not None:
            previous_cloud = _cloud_mask(previous_clpp)
        else:
            previous_cloud = np.isfinite(previous_values["CLP"]) & (previous_values["CLP"] != 0)
        for name in PRODUCT_NAMES:
            if name in previous_values:
                previous_valid[name] = ~_invalid_product(previous_values[name], name, previous_cloud)
    if previous_clpp is not None:
        previous_clpp = np.asarray(previous_clpp, dtype=np.float32)
        if previous_clpp.shape != updated_clpp.shape:
            raise ValueError(f"上一时刻 CLP_P 与当前网格不一致: {previous_clpp.shape} != {updated_clpp.shape}")
        previous_valid["CLP_P"] = ~_invalid_probability(previous_clpp)

    replacement_mask = np.zeros(updated["CLP"].shape, dtype=bool)
    bad_union = np.zeros(updated["CLP"].shape, dtype=bool)
    for invalid in current_invalid.values():
        bad_union |= invalid
    stats = {
        "quality_bad_pixels": int(bad_union.sum()),
        "replaced_pixels": 0,
        "unresolved_pixels": 0,
        "bad_by_product": {},
        "replaced_by_product": {},
        "unresolved_by_product": {},
    }

    for name in PRODUCT_NAMES:
        invalid = current_invalid[name]
        stats["bad_by_product"][name] = int(invalid.sum())
        updated[name][invalid] = np.nan
        if previous_values is not None and name in previous_values:
            previous = np.asarray(previous_values[name], dtype=np.float32)
            replace = invalid & previous_valid[name]
            updated[name][replace] = previous[replace]
            replacement_mask |= replace
            stats["replaced_by_product"][name] = int(replace.sum())
        else:
            stats["replaced_by_product"][name] = 0

    invalid_clpp = current_invalid["CLP_P"]
    stats["bad_by_product"]["CLP_P"] = int(invalid_clpp.sum())
    updated_clpp[:, invalid_clpp] = np.nan
    if previous_clpp is not None and "CLP_P" in previous_valid:
        previous_probabilities = np.asarray(previous_clpp, dtype=np.float32)
        replace_clpp = invalid_clpp & previous_valid["CLP_P"]
        updated_clpp[:, replace_clpp] = previous_probabilities[:, replace_clpp]
        replacement_mask |= replace_clpp
        stats["replaced_by_product"]["CLP_P"] = int(replace_clpp.sum())
    else:
        stats["replaced_by_product"]["CLP_P"] = 0

    stats["replaced_pixels"] = int(replacement_mask.sum())
    unresolved = np.zeros_like(replacement_mask)
    updated_cloud = _cloud_mask(updated_clpp)
    for name in PRODUCT_NAMES:
        product_unresolved = _invalid_product(updated[name], name, updated_cloud)
        unresolved |= product_unresolved
        stats["unresolved_by_product"][name] = int(product_unresolved.sum())
    unresolved_clpp = _invalid_probability(updated_clpp)
    unresolved |= unresolved_clpp
    stats["unresolved_by_product"]["CLP_P"] = int(unresolved_clpp.sum())
    stats["unresolved_pixels"] = int(unresolved.sum())
    return updated, updated_clpp, replacement_mask, stats


def apply_previous_global_continuous_fallback(
    current_values: Mapping[str, np.ndarray],
    current_clp: np.ndarray,
    previous_values: Mapping[str, np.ndarray] | None,
    previous_clp: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """在全球拼接完成后再次回填 CTH、CER 和 COT 缺失像元。

    单星质量控制在各卫星原始网格上执行，但 CLP 分类分数和三个连续产品采用
    不同的融合过程。重叠区融合后，最终 CLP 可能从晴空变成云，此时连续产品仍
    可能是 0；若只做拼接前检查，这类缺失会漏掉。因此全球结果写出前还需执行
    第二道质量控制。

    Parameters
    ----------
    current_values:
        当前时次全球 ``CTH``、``CER``、``COT`` 二维数组。
    current_clp:
        当前时次最终全球 CLP，0/1/2 分别表示晴空、液态云、冰云，未知为 NaN。
    previous_values:
        上一时次全球连续产品。上一全球文件不存在时传 ``None``。
    previous_clp:
        上一时次全球 CLP；用于保证只从上一时次真实云像元取值。

    Returns
    -------
    updated:
        回填后的三个连续产品。三者独立处理，某个产品缺失不会覆盖其他产品。
    replacement_mask:
        任一连续产品成功使用上一时次值的位置并集，应与已有 QA 做逻辑 OR。
    stats:
        按产品记录当前缺失数、成功回填数和仍未解决数，并给出像元并集统计。

    Notes
    -----
    晴空像元的连续产品 0 是业务约定，不参与上一时次回填；只有当前 CLP 为云且
    连续产品为 NaN/Inf、越界或低于云产品有效下限时才回填。若上一时次同一像元
    不是云或对应产品也无效，则当前值保持 NaN，QA 不置 1。
    """
    missing_names = [name for name in CLOUD_PRODUCTS if name not in current_values]
    if missing_names:
        raise ValueError(f"当前全球产品缺少变量: {missing_names}")

    clp = _validate_product_shape(current_clp, "CLP")
    expected_shape = clp.shape
    updated = {
        name: _validate_product_shape(current_values[name], name, expected_shape).copy()
        for name in CLOUD_PRODUCTS
    }
    current_cloud = np.isfinite(clp) & (clp != 0)

    previous_arrays: dict[str, np.ndarray] = {}
    previous_cloud = np.zeros(expected_shape, dtype=bool)
    if previous_values is not None and previous_clp is not None:
        previous_phase = _validate_product_shape(previous_clp, "previous_CLP", expected_shape)
        previous_cloud = np.isfinite(previous_phase) & (previous_phase != 0)
        for name in CLOUD_PRODUCTS:
            if name in previous_values:
                previous_arrays[name] = _validate_product_shape(
                    previous_values[name], f"previous_{name}", expected_shape
                )

    replacement_mask = np.zeros(expected_shape, dtype=bool)
    missing_union = np.zeros(expected_shape, dtype=bool)
    unresolved_union = np.zeros(expected_shape, dtype=bool)
    stats: dict[str, object] = {
        "missing_pixels": 0,
        "replaced_pixels": 0,
        "unresolved_pixels": 0,
        "missing_by_product": {},
        "replaced_by_product": {},
        "unresolved_by_product": {},
    }

    for name in CLOUD_PRODUCTS:
        # 只把云像元中的异常连续参数定义为“缺失”；晴空将在调用方统一写为 0。
        missing = current_cloud & _invalid_product(updated[name], name, current_cloud)
        missing_union |= missing
        updated[name][missing] = np.nan

        replace = np.zeros(expected_shape, dtype=bool)
        if name in previous_arrays:
            previous_valid = previous_cloud & ~_invalid_product(
                previous_arrays[name], name, previous_cloud
            )
            replace = missing & previous_valid
            updated[name][replace] = previous_arrays[name][replace]
            replacement_mask |= replace

        unresolved = current_cloud & _invalid_product(updated[name], name, current_cloud)
        unresolved_union |= unresolved
        stats["missing_by_product"][name] = int(missing.sum())
        stats["replaced_by_product"][name] = int(replace.sum())
        stats["unresolved_by_product"][name] = int(unresolved.sum())

    stats["missing_pixels"] = int(missing_union.sum())
    stats["replaced_pixels"] = int(replacement_mask.sum())
    stats["unresolved_pixels"] = int(unresolved_union.sum())
    return updated, replacement_mask, stats
