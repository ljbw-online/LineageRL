from multiprocessing import Process, Queue

import ale_py
import gymnasium as gym
import cv2

gym.register_envs(ale_py)  # IDE appeasement


# from environments.follow_the_dots import Env


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


def resize(im, width=None, height=None):
    if width is not None and height is not None:
        resize_width = width
        resize_height = height
    else:
        im_width = im.shape[1]
        im_height = im.shape[0]

        if width is not None:
            resize_width = width
            resize_height = int(im_height * (width / im_width))
        elif height is not None:
            resize_height = height
            resize_width = int(im_width * (height / im_height))
        else:
            raise ValueError('One of width or height has to be not None')

    return cv2.resize(
        im, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST
    )


def downsample_observation(observation):
    if observation.shape[0] > 84:
        observation = resize(observation, width=84, height=84)

    try:
        if observation.shape[2] == 3:
            return cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    except IndexError:
        pass

    return observation


def downscale(image):
    pass


def env_runner_fn(in_q, out_q):
    # env = Env(4)
    env = gym.make('ALE/Breakout-v5')

    out_q.put((env.observation_space.shape, env.action_space.n))

    # Wait for the first reset request
    in_q.get()

    while True:
        obs, _ = env.reset()
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

            obs, reward, terminal, _, _ = env.step(action)
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
    window = Window(image=obs)

    terminal = False
    while True:
        if terminal:
            sleep(1)
            window.update(out_q.get())
        elif step_count == max_ep_len:
            runner.reset()
            window.update(out_q.get())
            step_count = 0

        runner.put(int(input()))

        (obs, reward, terminal) = out_q.get()

        window.update(obs)

        step_count += 1
