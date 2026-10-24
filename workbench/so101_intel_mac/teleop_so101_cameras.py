"""SO101 teleoperation with both USB cameras and the Intel-compatible Rerun SDK.

Run this from Terminal. It enables physical follower movement.
Use --check to test display logging without connecting to cameras or motors.
"""
import sys
import rerun as rr

# LeRobot 0.4.4 expects Rerun >=0.24; Intel Mac uses 0.22 for NumPy 1.x.
# LeRobot logs individual scalar values, which the older Scalar API supports.
if not hasattr(rr, 'Scalars'):
    rr.Scalars = rr.Scalar


def main():
    if sys.argv[1:] == ['--check']:
        import numpy as np
        from lerobot.utils.visualization_utils import log_rerun_data
        rr.init('so101_camera_logging_check', spawn=False, strict=True)
        log_rerun_data(
            observation={'camera_0': np.zeros((480, 640, 3), dtype=np.uint8),
                         'camera_1': np.zeros((480, 640, 3), dtype=np.uint8),
                         'shoulder_pan.pos': 0.0},
            action={'shoulder_pan.pos': 0.0},
        )
        rr.rerun_shutdown()
        print('PASS: two camera images and joint/action scalar logging; no hardware connected.')
        return

    from lerobot.scripts.lerobot_teleoperate import main as teleoperate_main
    sys.argv = [sys.argv[0],
        '--robot.type=so101_follower',
        '--robot.port=/dev/cu.usbmodem5AE60801371',
        '--robot.id=follower',
        '--teleop.type=so101_leader',
        '--teleop.port=/dev/cu.usbmodem5AE60548171',
        '--teleop.id=leader',
        '--robot.cameras={camera_0: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera_1: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}',
        '--display_data=true',
        '--fps=30',
        *sys.argv[1:],
    ]
    teleoperate_main()


if __name__ == '__main__':
    main()
