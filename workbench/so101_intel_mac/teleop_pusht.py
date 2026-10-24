"""Mouse teleoperation of PushT through the installed LeRobot environment API.

Run: .venv/bin/python teleop_pusht.py
Check without opening a window: .venv/bin/python teleop_pusht.py --check
"""
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true', help='Run 30 automated steps without a desktop window')
args = parser.parse_args()
if args.check:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import pygame
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


def main():
    cfg = PushtEnv(render_mode='rgb_array', visualization_width=512, visualization_height=512)
    env = make_env(cfg, n_envs=1)['pusht'][0]
    try:
        pygame.display.init()
        window = pygame.display.set_mode((512, 512))
        pygame.display.set_caption('LeRobot PushT — hold left mouse to move | R reset | Esc quit')
        clock = pygame.time.Clock()
        observation, _ = env.reset(seed=42)
        action = observation['agent_pos'].copy().astype(np.float32)
        steps = 0
        running = True
        print('Hold left mouse button to move the blue pusher. Push the T onto the green target.')
        print('R: reset | Esc / close window: quit', flush=True)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    observation, _ = env.reset()
                    action = observation['agent_pos'].copy().astype(np.float32)
            if not running:
                break
            if args.check:
                action[0] = [256 + 100 * np.sin(steps / 8), 256 + 100 * np.cos(steps / 8)]
            elif pygame.mouse.get_pressed()[0] and pygame.mouse.get_focused():
                action[0] = np.clip(pygame.mouse.get_pos(), 0, 511)
            observation, reward, terminated, truncated, _ = env.step(action)
            batch = preprocess_observation(observation)
            assert batch[OBS_IMAGE].shape == (1, 3, 96, 96)
            assert batch[OBS_STATE].shape == (1, 2)
            frame = env.envs[0].render()
            window.blit(pygame.surfarray.make_surface(frame.transpose(1, 0, 2)), (0, 0))
            pygame.display.flip()
            if terminated[0] or truncated[0]:
                action = observation['agent_pos'].copy().astype(np.float32)
                print('Episode reset. Keep pushing, or press R for a new layout.', flush=True)
            steps += 1
            if args.check and steps >= 30:
                print('PASS: 30 physics steps, rendered frames, and LeRobot image/state tensor preprocessing.')
                break
            if not args.check:
                clock.tick(cfg.fps)
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    main()
