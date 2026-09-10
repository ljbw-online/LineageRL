from time import sleep
from multiprocessing import Process, Queue

import numpy as np

import sdl2
import sdl2.ext
from sdl2.ext.pixelaccess import pixels3d

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)  # IDE appeasement


def open_window():
    sdl2.ext.init()
    window = sdl2.ext.Window("SDL Window", size=(840, 840))
    surface = window.get_surface()
    view = pixels3d(surface)

    for i in range(840):
        view[i, i] = 255

    window.show()
    running = True
    while running:
        events = sdl2.ext.get_events()
        for event in events:
            match event.type:
                case sdl2.SDL_QUIT:
                    running = False
                    break
                case sdl2.SDL_TEXTINPUT:
                    # print(event.text.text.decode())
                    if event.text.text.decode() == 'q':
                        running = False
                        break

        window.refresh()

        # sleep(0) can cause CPU to spin up
        sleep(0.1)


# sdl2 appears to be using BGRA but the alpha channel is ignored.
# I was not able to find out how to get sdl2 to tell me which pixel
# format it is using.
def reformat_image(obs, scale):
    obs = np.repeat(obs, scale, axis=0)
    obs = np.repeat(obs, scale, axis=1)

    if obs.ndim == 3:
        # Swap width and height and reverse channels dimension
        return np.moveaxis(obs, 0, 1)[:, :, ::-1]
    else:
        rgb_obs = np.repeat(obs[:, :, None], 3, axis=2)
        return np.moveaxis(rgb_obs, 0, 1)


def play_env(env, key_map, scale=None):
    obs, _ = env.reset()

    height, width = obs.shape[:2]

    window = Window('SDL', (height, width), scale)
    window.update(obs)

    while True:
        keys = window.get_keys()

        if len(keys) > 0:
            if keys[0] == 'q':
                break

            action = key_map[keys[0]]
            obs, reward, terminated, _, _ = env.step(action)
            window.update(obs)
            print(reward)

            if terminated:
                print('terminated')
                window.update(env.reset())

        sleep(0.1)


def play_breakout():
    env = gym.make('ALE/Breakout-v5')

    key_map = {'left': 3, 'up': 1, 'right': 2, 'down': 0}

    play_env(env, key_map)


def display_observations(observations):
    window = Window(image=observations[0])

    for obs in observations:
        window.update(obs)
        keys = window.get_keys(block=True)

        if keys == ['q']:
            break


def handle_io(name, shape, scale, in_q, out_q):
    sdl2.ext.init()

    scaled_height, scaled_width = shape

    window = sdl2.ext.Window(name, size=(scaled_width, scaled_height))

    view = pixels3d(window.get_surface())

    window.show()
    # print('Created window with ID', sdl2.SDL_GetWindowID(window.window))

    running = True
    while running:
        if not in_q.empty():
            image = in_q.get()

            if image is None:
                break

            view[:, :, :3] = reformat_image(image, scale)

        window.refresh()

        events = sdl2.ext.get_events()
        for event in events:
            # print('Window', event.window.windowID, 'got event')
            if event.type == sdl2.SDL_QUIT:
                running = False
                out_q.put('q')
                break
            elif event.type == sdl2.SDL_KEYDOWN:
                match event.key.keysym.sym:
                    case sdl2.SDLK_q:
                        running = False
                        out_q.put('q')
                        break
                    case sdl2.SDLK_LEFT:
                        out_q.put('left')
                    case sdl2.SDLK_UP:
                        out_q.put('up')
                    case sdl2.SDLK_RIGHT:
                        out_q.put('right')
                    case sdl2.SDLK_DOWN:
                        out_q.put('down')
                    case sdl2.SDLK_RETURN:
                        out_q.put('enter')

        sleep(0.1)

    window.close()
    sdl2.ext.quit()
    # Possibly not necessary
    in_q.cancel_join_thread()
    out_q.cancel_join_thread()


class Window:
    def __init__(self, name='SDL', shape=None, scale=None, image=None):
        self.in_q = Queue()
        self.out_q = Queue()

        if shape is None:
            height = image.shape[0]
            width = image.shape[1]
        else:
            height, width = shape

        if scale is None:
            scale = int(500 / width)

        scaled_height = height * scale
        scaled_width = width * scale

        scaled_shape = (scaled_height, scaled_width)

        process = Process(
            target=handle_io,
            args=(name, scaled_shape, scale, self.in_q, self.out_q),
            daemon=True
        )

        process.start()

        if image is not None:
            self.update(image)

    def get_keys(self, block=False):
        keys = []

        if block:
            keys.append(self.out_q.get(block=True))
        else:
            while not self.out_q.empty():
                keys.append(self.out_q.get())

        return keys

    def update(self, image):
        self.in_q.put(image)

    def close(self):
        self.in_q.put(None)


if __name__ == '__main__':
    # from environments.random_grey_walls import Env
    # from networks import MLP

    # env = Env(size=9)

    # net = MLP(9, 4)

    # observations = env.get_renderable_q_values(net)

    # display_observations(observations[None, ...])

    # key_map = {'left': 0, 'up': 1, 'right': 2, 'down': 3}

    # play_env(env, key_map)

    play_breakout()
