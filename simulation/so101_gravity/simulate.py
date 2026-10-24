"""Offline full-arm SO101 rigid-body replay. No robot/serial imports or actuation."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KGCM_TO_NM = 0.0980665


@dataclass(frozen=True)
class Servo:
    name: str
    voltage_v: float
    stall_torque_kgcm: float
    no_load_s_per_60deg: float
    stall_current_a: float
    nominal_kt_kgcm_a: float

    @property
    def stall_nm(self):
        return self.stall_torque_kgcm * KGCM_TO_NM

    @property
    def speed_rad_s(self):
        return np.pi / 3 / self.no_load_s_per_60deg


SERVOS = (Servo("12v", 12, 30, 0.222, 2.7, 11), Servo("7p4v", 7.4, 19.5, 0.192, 2.5, 7.8))


def read_log(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    samples = [r for r in records if r.get("type") == "sample"]
    transition = next(
        r for r in samples if r.get("previous") == "lagging" and r.get("state") == "awaiting_alignment"
    )
    start = next(r["monotonic_s"] for r in samples if r.get("reason") == "keyboard_start")
    end = transition["monotonic_s"]
    samples = [r for r in samples if start <= r["monotonic_s"] < end]
    times = np.array([r["monotonic_s"] - start for r in samples])
    measured = np.array([[r["follower"][j] for j in JOINTS] for r in samples])
    command = np.array([[r["sent_action"][j + ".pos"] for j in JOINTS] for r in samples])
    if (
        len(times) < 3
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(measured).all()
        or not np.isfinite(command).all()
    ):
        raise ValueError("Invalid or insufficient replay samples")
    return times, measured, command, transition


def mapping(calibration, offsets_deg=None):
    # LeRobot examples pass arm degrees directly to this public new-calibration URDF.
    # Local range-midpoint zero is not independently aligned to CAD: expose offsets.
    scale = np.full(6, np.pi / 180)
    grip = calibration["gripper"]
    scale[-1] = (grip["range_max"] - grip["range_min"]) / 100 * 2 * np.pi / 4095
    offset = np.deg2rad(np.array(offsets_deg or [0] * 6, dtype=float))
    return scale, offset


def build_model(urdf, destination, servo, gravity, friction=0.0, timestep=0.001):
    tree = ET.parse(urdf)
    root = tree.getroot()
    # Meshes/contact deliberately removed; imported inertials and transforms retained.
    for link in root.findall("link"):
        for element in list(link):
            if element.tag in ("visual", "collision"):
                link.remove(element)
    for element in list(root):
        if element.tag in ("transmission", "material"):
            root.remove(element)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", {"fusestatic": "false", "discardvisual": "true"})
    initial = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    mujoco.mj_saveLastXML(str(destination), initial)
    model = mujoco.MjModel.from_xml_path(str(destination))
    model.opt.timestep = timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.gravity[:] = (0, 0, -9.81 if gravity else 0)
    ids = np.array([model.joint(j).dofadr[0] for j in JOINTS])
    if model.nv != 6 or list(ids) != list(range(6)):
        raise ValueError("Unexpected joint order; refusing implicit reordering")
    # DC back-EMF: tau = stall_torque * duty - (stall_torque / no_load_speed) * qdot.
    # Implicit joint damping integrates the electrical term stably for low link inertia.
    model.dof_damping[:] = servo.stall_nm / servo.speed_rad_s
    model.dof_frictionloss[:] = friction
    model.dof_armature[:] = 0.0001  # Assumed reflected rotor inertia, not manufacturer data.
    # Disable contact/limits for this free-space causality comparison. Original limits are
    # retained in XML; no claimed hard-stop simulation. Report excursions separately.
    model.jnt_limited[:] = 0
    mujoco.mj_saveLastXML(str(destination), model)
    return model


def pwm(error, velocity, p_register=16, d_register=32, derivative_sample_s=0.001, gain_scale=1):
    # Exploratory interpretation of documented P/4 and D/8 factors, not verified firmware.
    ticks_per_rad = 4095 / (2 * np.pi)
    p = p_register / 4 * error * ticks_per_rad
    d = d_register / 8 * (-velocity) * ticks_per_rad * derivative_sample_s
    return np.clip(gain_scale * (p + d) / 1000, -1, 1)


def simulate(model, servo, times, measured, command, scale, offset, gain_scale=1, derivative_sample_s=0.001):
    data = mujoco.MjData(model)
    data.qpos[:] = measured[0] * scale + offset
    data.qvel[:] = 0  # Initial measured velocities unavailable/uncertain.
    mujoco.mj_forward(model, data)
    positions, torques, duties = [], [], []
    dt = model.opt.timestep
    for k, output_time in enumerate(times):
        while data.time < output_time - dt / 2:
            index = max(0, np.searchsorted(times, data.time, side="right") - 1)
            goal = command[index] * scale + offset
            duty = pwm(
                goal - data.qpos, data.qvel, gain_scale=gain_scale, derivative_sample_s=derivative_sample_s
            )
            data.qfrc_applied[:] = servo.stall_nm * duty
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qpos)) > 20:
                raise RuntimeError("Divergent simulation")
        goal = command[k] * scale + offset
        duty = pwm(
            goal - data.qpos, data.qvel, gain_scale=gain_scale, derivative_sample_s=derivative_sample_s
        )
        positions.append((data.qpos.copy() - offset) / scale)
        torques.append(servo.stall_nm * duty - model.dof_damping * data.qvel)
        duties.append(duty)
    return np.asarray(positions), np.asarray(torques), np.asarray(duties)


def gravity_along_measurements(model, measured, scale, offset):
    data = mujoco.MjData(model)
    values = []
    for pose in measured:
        data.qpos[:] = pose * scale + offset
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        # qfrc_bias is torque needed to counter gravity at rest, not torque exerted by gravity.
        values.append(-data.qfrc_bias.copy())
    return np.asarray(values)


def metrics(times, measured, predicted, command):
    tail = times >= times[-1] - 4
    j = JOINTS.index("shoulder_lift")
    m, p, c = measured[tail, j], predicted[tail, j], command[tail, j]
    return {
        "shoulder_rmse_all_deg": float(np.sqrt(np.mean((predicted[:, j] - measured[:, j]) ** 2))),
        "shoulder_rmse_last4s_deg": float(np.sqrt(np.mean((p - m) ** 2))),
        "shoulder_measured_last4s_change_deg": float(m[-1] - m[0]),
        "shoulder_simulated_last4s_change_deg": float(p[-1] - p[0]),
        "shoulder_final_measured_deg": float(m[-1]),
        "shoulder_final_simulated_deg": float(p[-1]),
        "shoulder_final_command_deg": float(c[-1]),
        "shoulder_final_simulated_error_deg": float(p[-1] - c[-1]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--gain-scale", type=float, default=1.0)
    parser.add_argument("--derivative-sample-s", type=float, default=0.001)
    parser.add_argument("--friction-nm", type=float, default=0.0)
    parser.add_argument("--offsets-deg", type=float, nargs=6, default=[0.0] * 6)
    args = parser.parse_args()
    if (
        not all(
            np.isfinite(v)
            for v in [args.gain_scale, args.derivative_sample_s, args.friction_nm, *args.offsets_deg]
        )
        or args.gain_scale <= 0
        or args.derivative_sample_s <= 0
        or args.friction_nm < 0
    ):
        parser.error("Nonfinite/invalid model parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    urdf = here / "vendor/so101_new_calib.urdf"
    times, measured, command, transition = read_log(args.log)
    scale, offset = mapping(json.loads(args.calibration.read_text()), args.offsets_deg)
    cases, traces = {}, {}
    for servo in SERVOS:
        for gravity in (False, True):
            name = f"{servo.name}_gravity_{'on' if gravity else 'off'}"
            path = args.output / (name + ".xml")
            model = build_model(urdf, path, servo, gravity, friction=args.friction_nm)
            predicted, torque, duty = simulate(
                model,
                servo,
                times,
                measured,
                command,
                scale,
                offset,
                gain_scale=args.gain_scale,
                derivative_sample_s=args.derivative_sample_s,
            )
            cases[name] = {
                "servo": asdict(servo),
                "gravity": gravity,
                **metrics(times, measured, predicted, command),
            }
            traces[name] = predicted
            np.savez_compressed(
                args.output / (name + ".npz"),
                times_s=times,
                measured=measured,
                command=command,
                predicted=predicted,
                motor_shaft_torque_nm=torque,
                duty=duty,
            )
            # Companion config records external control law required with this passive MJCF.
            (args.output / (name + ".json")).write_text(
                json.dumps(
                    {
                        "servo": asdict(servo),
                        "gravity": gravity,
                        "p_register": 16,
                        "i_register": 0,
                        "d_register": 32,
                        "assumed_derivative_sample_s": args.derivative_sample_s,
                        "gain_scale": args.gain_scale,
                        "friction_nm": args.friction_nm,
                        "offsets_deg": args.offsets_deg,
                        "controller": "simulate.py:pwm",
                        "model_xml": path.name,
                    },
                    indent=2,
                )
                + "\n"
            )
            if gravity:
                gravity_torque = gravity_along_measurements(model, measured, scale, offset)
    np.savez_compressed(args.output / "measured_gravity.npz", times_s=times, gravity_torque_nm=gravity_torque)
    tail = times >= times[-1] - 4
    shoulder_gravity = gravity_torque[tail, 1]
    summary = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "settings": {
            "gain_scale": args.gain_scale,
            "derivative_sample_s": args.derivative_sample_s,
            "friction_nm": args.friction_nm,
            "offsets_deg": args.offsets_deg,
        },
        "cases": cases,
        "gravity_shoulder_last4s_nm": {
            "min": float(shoulder_gravity.min()),
            "max": float(shoulder_gravity.max()),
            "start": float(shoulder_gravity[0]),
            "end": float(shoulder_gravity[-1]),
        },
        "samples": len(times),
        "duration_s": float(times[-1]),
        "transition_reason": transition["reason"],
        "input_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.log, args.calibration, urdf)
        },
        "mapping": {
            "arm": "direct degrees -> radians; matches LeRobot kinematics example; not physically verified",
            "gripper": "calibrated 0..100 opening -> radians of measured encoder span; zero assumed closed",
            "scale_rad_per_unit": scale.tolist(),
            "offset_rad": offset.tolist(),
        },
        "limitations": [
            "nominal voltage models, not the logged approximately 5V supply",
            "PD register to duty law and derivative period are assumptions",
            "no collision, stiction default zero",
            "all six motors use one variant; actual SO101 may mix gear ratios",
            "zero initial velocity, assumed reflected rotor inertia .0001 kg m2",
            "no calibrated current sensor model, protection state machine or verified gain identification",
        ],
    }
    if args.sweep:
        sweeps = []
        # Last four seconds conditioned on measured starting pose, all six joints dynamic.
        st = np.flatnonzero(tail)[0]
        short_t = times[st:] - times[st]
        for servo in SERVOS:
            for gravity in (False, True):
                for gain in (0.5, 1, 2):
                    for shoulder_offset in (-5, 0, 5):
                        model = build_model(urdf, args.output / "sweep-model.xml", servo, gravity)
                        off = offset.copy()
                        off[1] += np.deg2rad(shoulder_offset)
                        pred, _, _ = simulate(
                            model, servo, short_t, measured[st:], command[st:], scale, off, gain_scale=gain
                        )
                        sweeps.append(
                            {
                                "servo": servo.name,
                                "gravity": gravity,
                                "gain_scale": gain,
                                "shoulder_offset_deg": shoulder_offset,
                                **metrics(short_t, measured[st:], pred, command[st:]),
                            }
                        )
        summary["sensitivity_last4s"] = sweeps
        (args.output / "sweep-model.xml").unlink()
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "comparison.csv").open("w") as f:
        columns = ["time_s", "measured_shoulder_deg", "command_shoulder_deg", *traces]
        writer = csv.writer(f)
        writer.writerow(columns)
        for k, t in enumerate(times):
            writer.writerow([t, measured[k, 1], command[k, 1], *[p[k, 1] for p in traces.values()]])
    plot(args.output, times, measured, command, traces, gravity_torque)
    print(
        json.dumps(
            {"cases": cases, "gravity_shoulder_last4s_nm": summary["gravity_shoulder_last4s_nm"]}, indent=2
        )
    )


def plot(output, times, measured, command, traces, gravity_torque):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for ax in axes[:2]:
        ax.plot(times, measured[:, 1], "k", lw=2, label="Measured shoulder")
        ax.plot(times, command[:, 1], "k--", lw=1, label="Sent shoulder goal")
        for name, pred in traces.items():
            ax.plot(times, pred[:, 1], label=name.replace("_", " "), alpha=0.85)
        ax.set_ylabel("Shoulder lift (degrees)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlim(times[-1] - 4, times[-1])
    axes[1].set_title("Final four seconds before disengagement")
    tail = times >= times[-1] - 4
    visible = np.concatenate([measured[tail, 1], command[tail, 1], *[p[tail, 1] for p in traces.values()]])
    axes[1].set_ylim(visible.min() - 1, visible.max() + 1)
    axes[2].plot(times, gravity_torque[:, 1], color="tab:purple")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_ylabel("Gravity torque on shoulder (Nm)")
    axes[2].set_xlabel("Seconds since Start")
    axes[2].grid(alpha=0.25)
    fig.suptitle("SO101 nominal servo comparison — exploratory, unvalidated actuator/coordinate mapping")
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
