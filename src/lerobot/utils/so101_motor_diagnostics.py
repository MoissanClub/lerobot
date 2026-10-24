"""Read-only, timestamped SO101 motor registers for supervised inspection.

Values are raw register units after the motor driver's sign decoding. They are
not calibrated amperes, load percentages, or a decoded protection diagnosis.
"""

import time

DYNAMIC_REGISTERS = (
    "Torque_Enable",
    "Present_Current",
    "Present_Load",
    "Present_Temperature",
    "Present_Voltage",
    "Present_Velocity",
    "Status",
    "Moving",
)
CONFIG_REGISTERS = (
    "Operating_Mode",
    "P_Coefficient",
    "I_Coefficient",
    "D_Coefficient",
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Homing_Offset",
    "Max_Torque_Limit",
    "Torque_Limit",
    "Protection_Current",
    "Max_Temperature_Limit",
    "Unloading_Condition",
    "Overload_Torque",
    "Goal_Velocity",
    "Acceleration",
)


class MotorDiagnostics:
    """Use the existing bus on its owning thread; never create a serial reader thread."""

    def __init__(self, bus, interval_s=0.15, clock=None):
        self.bus = bus
        self.interval_s = interval_s
        self.clock = clock or time.monotonic
        self.latest = {}
        self.index = 0
        self.next_read = self.clock() + interval_s

    def read(self, register):
        start = self.clock()
        try:
            values = self.bus.sync_read(register, normalize=False, num_retry=0)
            values = {key: int(value) for key, value in values.items()}
            error = None
        except Exception as exc:
            values = None
            error = repr(exc)
        end = self.clock()
        result = {
            "read_started_monotonic_s": start,
            "read_finished_monotonic_s": end,
            "duration_s": end - start,
            "values": values,
            "error": error,
        }
        self.latest[register] = result
        return result

    def snapshot(self):
        return {register: self.read(register) for register in (*CONFIG_REGISTERS, *DYNAMIC_REGISTERS)}

    def poll(self, now):
        """At most one optional diagnostic read per control iteration."""
        if now < self.next_read:
            return None
        register = DYNAMIC_REGISTERS[self.index % len(DYNAMIC_REGISTERS)]
        result = self.read(register)
        self.index += 1
        self.next_read = self.clock() + self.interval_s
        return {"register": register, **result}


def diagnostic_value(latest, register, joint):
    record = latest.get(register)
    if not record or record["error"] is not None or record["values"] is None:
        return None
    return record["values"].get(joint)
