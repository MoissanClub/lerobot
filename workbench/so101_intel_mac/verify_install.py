"""Check local LeRobot imports and CLI help without accessing robot hardware."""
import importlib
import importlib.metadata
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parent
for name in ('numpy', 'torch', 'torchvision', 'av', 'cv2', 'datasets', 'rerun', 'lerobot',
             'lerobot.datasets.lerobot_dataset', 'lerobot.policies.act.modeling_act'):
    importlib.import_module(name)
    print(f'PASS import {name}', flush=True)
import torch
assert torch.equal(torch.ones(2, 2) @ torch.ones(2, 2), torch.full((2, 2), 2.0))
print(f'PASS CPU tensor calculation; MPS={torch.backends.mps.is_available()}; CUDA={torch.cuda.is_available()}', flush=True)
for package in ('lerobot', 'torch', 'torchvision', 'numpy', 'rerun-sdk', 'av'):
    print(f'{package}=={importlib.metadata.version(package)}', flush=True)
failures = []
for command in ('lerobot-info', 'lerobot-train', 'lerobot-eval', 'lerobot-record',
                'lerobot-replay', 'lerobot-calibrate', 'lerobot-teleoperate', 'lerobot-find-cameras'):
    args = [str(pathlib.Path(sys.executable).parent / command)]
    if command != 'lerobot-info':
        args.append('--help')
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=90)
        (root / f'{command}.log').write_text(result.stdout + result.stderr)
        print(f'{"PASS" if result.returncode == 0 else "FAIL"} {command} (exit {result.returncode})', flush=True)
        if result.returncode:
            failures.append(command)
    except subprocess.TimeoutExpired:
        failures.append(command)
        print(f'FAIL {command}: timeout', flush=True)
if failures:
    raise SystemExit('CLI failures: ' + ', '.join(failures))
