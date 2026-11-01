"""V2 输出策略的统一定义。

``skip``、``overwrite`` 和 ``repair`` 只决定是否需要重新计算，不改变文件
写出方式。所有正式文件仍然必须先写临时文件、校验成功后再原子替换。
"""

from typing import Optional


OUTPUT_POLICIES = ("skip", "overwrite", "repair")


def normalize_output_policy(value: Optional[str], default: str = "skip") -> str:
    """规范化输出策略，并对拼写错误尽早报错。"""
    policy = (value or default).strip().lower()
    if policy not in OUTPUT_POLICIES:
        raise ValueError(
            f"无效输出策略 {value!r}，可选值为: {', '.join(OUTPUT_POLICIES)}"
        )
    return policy


def is_overwrite_policy(value: Optional[str]) -> bool:
    """判断策略是否要求无条件重算并替换已有结果。"""
    return normalize_output_policy(value) == "overwrite"

