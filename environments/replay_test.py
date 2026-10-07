import numpy as np
from gymnasium.spaces import Box, Discrete


class Env:
    def __init__(self):
        self.size = 2

        self.episode_count = None
        self.step_count = 0

        self.ep_length = None

        self.observation_space = Box(0, 255, (2, 2), np.uint8)
        self.action_space = Discrete(4)

    def reset(self, length=10):
        self.step_count = 1
        self.ep_length = length

        obs = np.zeros((self.size, self.size), dtype=np.uint8)

        # First reset
        if self.episode_count is None:
            self.episode_count = 0
            obs[1, 0] = 255
        else:
            self.episode_count += 1

        obs[0, 0] = self.episode_count

        return obs, {}

    def step(self, action):
        if self.ep_length is None:
            raise ValueError

        obs = np.zeros((self.size, self.size), dtype=np.uint8)

        obs[0, 0] = self.episode_count
        obs[0, 1] = self.step_count
        obs[1, 0] = action

        self.step_count += 1

        term = self.step_count == self.ep_length

        reward = np.float32(term)

        obs[1, 1] = reward

        return obs, reward, term, False, {}


if __name__ == '__main__':
    env = Env()

    print(env.reset())

    while True:
        obs, reward, term, _, _ = env.step(int(input('action: ')))
        print(obs, reward, term)
        if term:
            print('reset')
            print(env.reset())
