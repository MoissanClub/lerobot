from lerobot.utils.so101_motor_diagnostics import DYNAMIC_REGISTERS
from lerobot.utils.so101_motor_history import MotorStateHistory


def test_three_second_window_preserves_values_and_errors_without_extra_trigger_reads():
    now = [0.0]

    class Bus:
        calls = 0

        def sync_read(self, register, **kwargs):
            self.calls += 1
            assert kwargs == {"normalize": False, "num_retry": 0}
            if register == "Present_Current":
                raise OSError("test missing register")
            return {"elbow_flex": 152}

    bus = Bus()
    history = MotorStateHistory(bus, clock=lambda: now[0])
    for k in range(101):
        now[0] = k * 0.1
        history.poll(now[0])
    calls = bus.calls
    snapshot = history.snapshot(10.0)
    assert bus.calls == calls  # Capturing history never touches serial I/O.
    assert not snapshot["startup_truncated"]
    assert snapshot["window_start_monotonic_s"] == 7
    assert all(7 <= r["read_started_monotonic_s"] <= 10 for r in snapshot["records"])
    assert {r["register"] for r in snapshot["records"]} == set(DYNAMIC_REGISTERS)
    assert any(r["error"] and r["values"] is None for r in snapshot["records"])
    assert any(r["values"] == {"elbow_flex": 152} for r in snapshot["records"])


def test_history_marks_short_startup_and_repeated_snapshots_are_independent():
    class Bus:
        def sync_read(self, *args, **kwargs):
            return {"elbow_flex": 1}

    now = [0.0]
    history = MotorStateHistory(Bus(), clock=lambda: now[0])
    now[0] = 0.1
    history.poll(0.1)
    first = history.snapshot(0.2)
    assert first["startup_truncated"] and len(first["records"]) == 1
    assert history.snapshot(4)["records"] == []
    assert len(first["records"]) == 1
