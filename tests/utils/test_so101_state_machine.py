import math

import pytest

from lerobot.utils.so101_state_machine import Config, Controller, State


def make():
    cfg = Config(
        alignment={"j": 0.2},
        lag={"j": 1},
        speed={"j": 2},
        acceleration={"j": 4},
        dwell_s=0.2,
        lag_dwell_s=0.2,
        progress_window_s=0.3,
        nonprogress_s=0.4,
        min_progress=0.05,
        max_feedback_gap_s=0.15,
        return_timeout_s=2,
    )
    return Controller({"j": 0}, {"j": (-10, 10)}, cfg)


def test_start_catchup_tracking_and_stop_with_fake_motor():
    c = make()
    actual = {"j": 0.0}
    transitions = []
    for k in range(250):
        events = ["start"] if k == 1 else ["stop"] if k == 150 else []
        d = c.step(k * 0.02, {"j": 2}, actual, events)
        if d.state != d.previous:
            transitions.append(d.state)
        if d.command is not None:
            actual = d.command.copy()  # Ideal position servo; controller drives this mock.
    assert transitions == ["idle", "lagging", "tracking", "returning_to_rest", "idle"]
    assert abs(actual["j"]) <= 0.2


@pytest.mark.parametrize("state", [State.TRACKING, State.LAGGING])
@pytest.mark.parametrize("q,t,reason", [(9.8, 11, "upper:j"), (-9.8, -11, "lower:j")])
def test_boundary_disengages_before_outward_command(state, q, t, reason):
    c = make()
    c.state = state
    d = c.step(1, {"j": t}, {"j": q})
    assert d.state == "awaiting_alignment"
    assert d.reason == reason
    assert d.command == {"j": q}


def test_inward_boundary_command_does_not_disengage():
    c = make()
    c.state = State.TRACKING
    assert c.step(1, {"j": 8}, {"j": 9.8}).state == "tracking"


def test_error_requires_lag_then_nonprogress_and_pickup_dwell():
    c = make()
    c.state = State.TRACKING
    states = []
    for k in range(25):
        d = c.step(k * 0.05, {"j": 4}, {"j": 0})
        states.append(d.state)
    assert states.index("lagging") < states.index("awaiting_alignment")
    assert d.command == {"j": 0}
    for k in range(25, 32):
        d = c.step(k * 0.05, {"j": 0.1}, {"j": 0})
    assert d.state == "tracking"
    assert abs(d.command["j"]) < 0.1


def test_fault_priority_stale_feedback_and_latched_halt():
    c = make()
    c.step(0, {"j": 0}, {"j": 0})
    d = c.step(1, {"j": 11}, {"j": 9.9}, ["stop", "halt"])
    assert d.state == "fault" and d.command is None
    assert c.step(1.05, {"j": 0}, {"j": 0}, ["start"]).state == "fault"
    c = make()
    d = c.step(0, {"j": 1}, {"j": 0}, ["halt", "stop"])
    assert d.state == "halted" and d.command == {"j": 0}
    assert c.step(0.05, {"j": 2}, {"j": 0}, ["start"]).command is None


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_nonfinite_feedback_never_produces_command(bad):
    assert make().step(0, {"j": bad}, {"j": 0}).command is None


def test_setup_requires_stop_to_return_and_does_not_start_from_arbitrary_pose():
    c = make()
    assert c.step(0, {"j": 8}, {"j": 3}, ["start"]).state == "setup_hold"
    assert c.step(0.05, {"j": 8}, {"j": 3}, ["stop"]).state == "returning_to_rest"
    for k in range(2, 30):
        c.step(k * 0.05, {"j": 8}, {"j": 3})
    assert c.state == State.FAULT


def test_command_slew_and_acceleration_on_moving_and_reversing_target():
    c = make()
    c.state = State.TRACKING
    commands = []
    # Large lag thresholds isolate trajectory shaping from supervisory transitions.
    c.cfg.lag["j"] = 50
    for k in range(100):
        d = c.step(k * 0.02, {"j": 8 if k < 50 else -8}, {"j": 0})
        commands.append(d.command["j"])
    velocities = [(b - a) / 0.02 for a, b in zip(commands, commands[1:], strict=False)]
    assert max(map(abs, velocities)) <= 2 + 1e-8
    assert max(abs(b - a) / 0.02 for a, b in zip(velocities, velocities[1:], strict=False)) <= 4 + 1e-8


def test_validation_rejects_inconsistent_thresholds_and_rest():
    c = make()
    c.cfg.lag["j"] = 0.1
    with pytest.raises(ValueError):
        Controller({"j": 0}, {"j": (-10, 10)}, c.cfg)
    with pytest.raises(ValueError):
        Controller({"j": 11}, {"j": (-10, 10)}, make().cfg)


def test_pickup_never_resumes_into_same_boundary_violation():
    c = make()
    c.cfg.alignment["j"] = 1
    c.state = State.TRACKING
    for k in range(30):
        d = c.step(k * 0.05, {"j": 10.2}, {"j": 10})
    assert d.state == "awaiting_alignment"


def test_pickup_first_command_is_exact_measured_position():
    c = make()
    c.state = State.ALIGN
    for k in range(10):
        d = c.step(k * 0.05, {"j": 0.1}, {"j": 0})
        if d.previous == "awaiting_alignment" and d.state == "tracking":
            assert d.command == {"j": 0}
            return
    pytest.fail("Did not resume")
