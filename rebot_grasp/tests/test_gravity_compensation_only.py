from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gravity_compensation_only as gravity_module  # noqa: E402
from gravity_compensation_only import (  # noqa: E402
    GRAVITY_KD,
    GRAVITY_KP,
    GRAVITY_TAU_SCALE,
    JOINT_DIRECTION,
    GravityCompensationOnly,
    angles_near_reference,
)


def test_parameters_match_ros2_gravity_compensation_config() -> None:
    np.testing.assert_array_equal(GRAVITY_KP, np.full(6, 2.0))
    np.testing.assert_array_equal(GRAVITY_KD, np.full(6, 1.5))
    np.testing.assert_array_equal(JOINT_DIRECTION, np.ones(6))
    np.testing.assert_array_equal(GRAVITY_TAU_SCALE, np.ones(6))


def test_gravity_command_uses_same_model_and_motor_direction() -> None:
    controller = GravityCompensationOnly()
    tau_model, tau_motor = controller._compute_command(np.zeros(6))
    np.testing.assert_allclose(tau_motor, tau_model)
    assert np.all(np.isfinite(tau_motor))


def test_encoder_wrap_is_continuous() -> None:
    previous = np.array([np.pi - 0.01, 0.0])
    wrapped = np.array([-np.pi + 0.01, 0.1])
    np.testing.assert_allclose(
        angles_near_reference(wrapped, previous),
        [np.pi + 0.01, 0.1],
        atol=1e-12,
    )


def test_initial_feedback_retries_until_all_six_motors_reply(
    monkeypatch,
) -> None:
    """回归：connect 后第一轮尚无状态时，不能立刻误报六轴掉线。"""

    class FakeMotor:
        def __init__(self, position: float) -> None:
            self.position = position
            self.request_count = 0

        def request_feedback(self) -> None:
            self.request_count += 1

        def get_state(self):
            if self.request_count < 2:
                return None
            return SimpleNamespace(pos=self.position)

    class FakeController:
        def poll_feedback_once(self) -> None:
            pass

    controller = GravityCompensationOnly()
    motors = {
        joint.name: FakeMotor(float(index))
        for index, joint in enumerate(controller.arm._jcfgs)
    }
    controller.robot._motor_map = motors
    controller.robot._ctrl_map = {"damiao": FakeController()}
    monkeypatch.setattr(gravity_module.time, "sleep", lambda _seconds: None)

    positions = controller._wait_for_initial_feedback()

    np.testing.assert_array_equal(positions, np.arange(6, dtype=np.float64))
    assert all(motor.request_count == 2 for motor in motors.values())
