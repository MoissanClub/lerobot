from unittest.mock import Mock

import pytest
import scservo_sdk as scs

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def make_bus(replies):
    bus = FeetechMotorsBus("/dev/null", {"joint": Motor(2, "sts3215", MotorNormMode.DEGREES)})
    bus.packet_handler = Mock()
    bus.packet_handler.writeTxRx.side_effect = replies
    bus.packet_handler.getTxRxResult.return_value = "Incorrect status packet"
    bus.packet_handler.getRxPacketError.return_value = "Servo fault"

    # Avoid serial I/O while retaining the real SDK communication-retry loop.
    def write(name, motor, value, num_retry=0):
        bus._write(55 if name == "Lock" else 40, 1, 2, value, num_retry=num_retry, err_msg=name)

    bus.write = write
    return bus


def test_lock_recovers_from_one_bad_reply_without_reenabling_torque():
    bus = make_bus([(scs.COMM_SUCCESS, 0), (scs.COMM_RX_CORRUPT, 0), (scs.COMM_SUCCESS, 0)])
    bus.enable_torque()
    calls = bus.packet_handler.writeTxRx.call_args_list
    assert [c.args[2] for c in calls] == [40, 55, 55]


def test_lock_failure_still_aborts_after_three_attempts():
    bus = make_bus([(scs.COMM_SUCCESS, 0)] + [(scs.COMM_RX_CORRUPT, 0)] * 3)
    with pytest.raises(ConnectionError):
        bus.enable_torque()
    assert bus.packet_handler.writeTxRx.call_count == 4


def test_motor_reported_fault_is_not_retried_as_bad_communication():
    bus = make_bus([(scs.COMM_SUCCESS, 0), (scs.COMM_SUCCESS, 4)])
    with pytest.raises(RuntimeError):
        bus.enable_torque()
    assert bus.packet_handler.writeTxRx.call_count == 2
