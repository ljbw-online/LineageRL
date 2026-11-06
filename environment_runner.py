from multiprocessing import Process, Queue

import numpy as np
import ale_py
import gymnasium as gym
import cv2

from common import Worker
# from environments.follow_the_dots import Env
# from environments.gta import Env

gym.register_envs(ale_py)  # IDE appeasement


def resize_image(im):
    return cv2.resize(im, (84, 84), interpolation=cv2.INTER_NEAREST)


def convert_to_greyscale(image):
    # I think the emulator actually produces RGB images, contrary to my
    # previous belief.
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def downscale(image):
    image = convert_to_greyscale(image)
    return resize_image(image)


def reset_stack(env):
    stack = np.empty((84, 84, 4), dtype=np.uint8)

    stack[:, :, 0] = downscale(env.reset()[0])

    stack[:, :, 1] = downscale(env.step(0)[0])
    stack[:, :, 2] = downscale(env.step(0)[0])
    stack[:, :, 3] = downscale(env.step(0)[0])

    return stack


# TODO: Downscaling should happen on the CPU but could we stack frames inside
# JAX?
def step_stack(env, action):
    stack = np.empty((84, 84, 4), dtype=np.uint8)

    stack_reward = 0.0
    stack_terminal = False

    for i in range(4):
        obs, reward, terminal, _, _ = env.step(action)

        stack[:, :, i] = downscale(obs)
        stack_reward += reward
        stack_terminal = stack_terminal or terminal

    stack_reward = np.clip(stack_reward, a_min=-1.0, a_max=1.0)

    return stack, stack_reward, stack_terminal


def stack_to_flat(stack):
    # Switch to channels-first
    flat = np.moveaxis(obs, 2, 0)
    return np.hstack(list(flat))


def env_runner_fn(in_q, out_q):
    # env = Env(4)
    env = gym.make('ALE/Breakout-v5')
    # env = Env()

    # out_q.put((env.observation_space.shape, env.action_space.n))
    out_q.put(((84, 84, 4), 4))

    # Wait for the first reset request
    in_q.get()

    while True:
        # obs, _ = env.reset()
        obs = reset_stack(env)
        out_q.put(obs)
        terminal = False
        while True:
            if terminal:
                in_q.get()  # Wait for reset request
                break

            action = in_q.get()

            # Reset requested
            if action is None:
                break

            # obs, reward, terminal, _, _ = env.step(action)
            obs, reward, terminal = step_stack(env, action)

            out_q.put((obs, reward, terminal))


class EnvRunner(Worker):
    def __init__(self, out_q=None):
        super().__init__(env_runner_fn, out_q=out_q)

        obs_shape, num_actions = self.get()

    def reset(self):
        self.put(None)
        return self.get()

    def step(self, action):
        self.put(action)
        return self.get()


if __name__ == '__main__':
    from time import sleep
    from visualisation import Window

    from multiprocessing import set_start_method
    set_start_method('spawn')

    max_ep_len = 30
    step_count = 0

    runner = EnvRunner()

    obs = runner.reset()

    window = Window(image=stack_to_flat(obs))

    terminal = False
    while True:
        if terminal:
            sleep(1)
        elif step_count == max_ep_len:
            print('truncating')
            obs = runner.reset()
            window.update(stack_to_flat(obs))
            step_count = 0

        obs, reward, terminal = runner.step(int(input()))

        window.update(stack_to_flat(obs))

        step_count += 1
