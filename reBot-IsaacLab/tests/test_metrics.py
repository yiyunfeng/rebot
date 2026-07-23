"""不启动 Isaac Sim 即可运行的成功率统计单元测试。"""

import pytest

from rebot_isaaclab.metrics import wilson_interval


def test_wilson_interval_contains_measured_rate():
    """常规样本下，置信区间应包含观测成功率且不越过概率边界。"""
    low, high = wilson_interval(750, 1000)
    assert low < 0.75 < high
    assert 0.0 <= low <= high <= 1.0


@pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (1, -1)])
def test_wilson_interval_rejects_invalid_counts(successes, total):
    """拒绝负计数和成功次数大于总次数的非法输入。"""
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_wilson_interval_handles_empty_sample():
    """尚无完成 episode 时返回稳定的零区间，不发生除零错误。"""
    assert wilson_interval(0, 0) == (0.0, 0.0)
