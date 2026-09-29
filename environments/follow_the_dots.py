import numpy as np

rng = np.random.default_rng()


class Env:
    def __init__(self, size, safe=True):
        self.size = size
        self.safe = safe

        if not safe:
            raise NotImplementedError

        self.x = 0
        self.y = 0

        self.dot_xs = rng.integers(size, size=size)

    def reset(self):
        self.x = 0
        self.y = 0

        obs = np.zeros((self.size, self.size), dtype=np.uint8)

        self.dot_xs = rng.integers(self.size, size=self.size)

        obs[np.arange(self.size), self.dot_xs] = 128
        obs[self.y, self.x] = 255

        return obs, {}

    def step(self, action):
        on_dot = self.x == self.dot_xs[self.y]

        match action:
            case 0:
                self.x -= 1
            case 1:
                self.y -= 1
            case 2:
                self.x += 1
            case 3:
                self.y += 1

        if self.safe:
            # Have we gone off the bottom at the x coord of the last dot
            term = self.y == self.size and self.x == self.dot_xs[-1]
            blocked = (not on_dot) and (action == 1 or action == 3)

            undo_action = (
                self.y < 0 or self.y >= self.size
                or self.x < 0 or self.x >= self.size
                or blocked
            )

            if undo_action:
                match action:
                    case 0:
                        self.x += 1
                    case 1:
                        self.y += 1
                    case 2:
                        self.x -= 1
                    case 3:
                        self.y -= 1

            obs = np.zeros((self.size, self.size), dtype=np.uint8)

            obs[np.arange(self.size), self.dot_xs] = 128

            if not term:
                obs[self.y, self.x] = 255

            reward = np.float32(term)

            return obs, reward, term, False, {}


if __name__ == '__main__':
    env = Env(8)

    print(env.reset())

    while True:
        print(env.step(int(input())))
