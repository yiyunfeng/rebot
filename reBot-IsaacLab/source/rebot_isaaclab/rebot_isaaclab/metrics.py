"""Pure-Python evaluation helpers; safe to test without launching Isaac Sim."""

import math


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """计算二项成功率的 95% Wilson 置信区间。

    Wilson 区间在成功率接近 0/1 或样本量较小时，比简单的正态近似更可靠。
    返回 ``(下界, 上界)``，两端均为 0~1 范围内的比例。

    Args:
        successes: 成功 episode 数。
        total: 已完成 episode 总数。

    Raises:
        ValueError: 计数为负数，或成功数大于总数。
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("必须满足 0 <= successes <= total")
    if total == 0:
        return 0.0, 0.0

    # 标准正态分布双侧 95% 分位数。
    z = 1.959963984540054
    rate = successes / total
    # Wilson score interval 的中心和半径；与 p ± z*sqrt(p(1-p)/n)
    # 不同，它通过 z²/n 修正有限样本偏差。
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius
