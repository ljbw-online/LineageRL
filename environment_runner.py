from multiprocessing import Process, Queue

import numpy as np
import ale_py
import gymnasium as gym
import cv2

# from environments.follow_the_dots import Env

gym.register_envs(ale_py)  # IDE appeasement


# def get_env_runner_fn(constructor):
#     def env_runner(in_q, out_q):
#         env = constructor()

#         # Wait for the first reset request
#         in_q.get()

#         while True:
#             obs, _ = env.reset()
#             out_q.put(obs)
#             terminal = False
#             while True:
#                 if terminal:
#                     break

#                 action = in_q.get()

#                 # Reset requested
#                 if action is None:
#                     break

#                 obs, reward, terminal, _, _ = env.step(action)
#                 out_q.put((obs, reward, terminal))

#     return env_runner


# def resize(im, width=None, height=None):
#     if width is not None and height is not None:
#         resize_width = width
#         resize_height = height
#     else:
#         im_width = im.shape[1]
#         im_height = im.shape[0]

#         if width is not None:
#             resize_width = width
#             resize_height = int(im_height * (width / im_width))
#         elif height is not None:
#             resize_height = height
#             resize_width = int(im_width * (height / im_height))
#         else:
#             raise ValueError('One of width or height has to be not None')

#     return cv2.resize(
#         im, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST
#     )


def resize_image(im):
    return cv2.resize(im, (84, 84), interpolation=cv2.INTER_NEAREST)


def convert_to_greyscale(image):
    # return cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

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


def env_runner_fn(in_q, out_q):
    # env = Env(4)
    env = gym.make('ALE/Breakout-v5')

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


class EnvRunner:
    def __init__(self, out_q):
        self.in_q = Queue()
        self.put = self.in_q.put

        # env_runner_fn = get_env_runner_fn(env_constructor)

        self.process = Process(
            target=env_runner_fn,
            args=(self.in_q, out_q),
            daemon=True
        )

        self.process.start()

    def reset(self):
        self.in_q.put(None)


def stack_to_flat(stack):
    # Switch to channels-first
    flat = np.moveaxis(obs, 2, 0)
    return np.hstack(list(flat))


if __name__ == '__main__':
    from time import sleep
    from visualisation import Window

    from multiprocessing import set_start_method
    set_start_method('spawn')

    max_ep_len = 30
    step_count = 0

    out_q = Queue()
    runner = EnvRunner(out_q)

    out_q.get()  # obs_shape and num_actions

    runner.reset()
    obs = out_q.get()

    window = Window(image=stack_to_flat(obs))

    terminal = False
    while True:
        if terminal:
            sleep(1)
            window.update(out_q.get())
        elif step_count == max_ep_len:
            print('truncating')
            runner.reset()
            window.update(stack_to_flat(out_q.get()))
            step_count = 0

        runner.put(int(input()))

        (obs, reward, terminal) = out_q.get()

        window.update(stack_to_flat(obs))

        step_count += 1
