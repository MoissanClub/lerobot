"""Motor state inspection must only read registers and preserve errors/units."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from lerobot.utils.so101_motor_diagnostics import (
    CONFIG_REGISTERS,
    DYNAMIC_REGISTERS,
    MotorDiagnostics,
    diagnostic_value,
)


def test_raw_register_read_does_not_normalize_or_write():
    bus = MagicMock()
    bus.sync_read.return_value = {"elbow_flex": -17}
    clock = iter([0, 1, 1.01])
    reader = MotorDiagnostics(bus, clock=lambda: next(clock))
    result = reader.read("Present_Current")
    bus.sync_read.assert_called_once_with("Present_Current", normalize=False, num_retry=0)
    bus.sync_write.assert_not_called()
    bus.write.assert_not_called()
    assert result["values"] == {"elbow_flex": -17}
    assert result["error"] is None
    assert diagnostic_value(reader.latest, "Present_Current", "elbow_flex") == -17


def test_failed_register_is_unavailable_not_zero():
    bus = MagicMock()
    bus.sync_read.side_effect = OSError("unsupported register or read failure")
    reader = MotorDiagnostics(bus)
    result = reader.read("Status")
    assert result["values"] is None
    assert "read failure" in result["error"]
    assert diagnostic_value(reader.latest, "Status", "elbow_flex") is None


def test_poll_reads_at_most_one_register_and_respects_interval():
    now = [0.0]
    bus = MagicMock()
    bus.sync_read.return_value = {"elbow_flex": 1}
    reader = MotorDiagnostics(bus, interval_s=0.15, clock=lambda: now[0])
    assert reader.poll(0.1) is None
    now[0] = 0.2
    assert reader.poll(0.2)["register"] == DYNAMIC_REGISTERS[0]
    assert reader.poll(0.2) is None
    now[0] = 0.4
    assert reader.poll(0.4)["register"] == DYNAMIC_REGISTERS[1]
    assert bus.sync_read.call_count == 2


def test_snapshot_captures_success_and_failure_for_every_register():
    calls = []

    def read(register, **kwargs):
        calls.append(register)
        if register == "Present_Load":
            raise ValueError("synthetic load failure")
        return {"elbow_flex": 16}

    reader = MotorDiagnostics(SimpleNamespace(sync_read=read))
    result = reader.snapshot()
    assert calls == [*CONFIG_REGISTERS, *DYNAMIC_REGISTERS]
    assert result["P_Coefficient"]["values"]["elbow_flex"] == 16
    assert result["Present_Load"]["error"] is not None
    assert result["Present_Load"]["values"] is None
