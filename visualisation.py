from time import sleep

import numpy as np

import sdl2
import sdl2.ext
from sdl2.ext.pixelaccess import pixels3d

import ale_py
import gymnasium as gym

from common import Worker


gym.register_envs(ale_py)  # IDE appeasement


# sdl2 appears to be using BGRA but the alpha channel is ignored.
# I was not able to find out how to get sdl2 to tell me which pixel
# format it is using.
def reformat_image(obs, bgra):
    # obs = np.repeat(obs, scale, axis=0)
    # obs = np.repeat(obs, scale, axis=1)

    if bgra:
        # If obs is already BGR then only swap width and height
        return np.moveaxis(obs, 0, 1)
    elif obs.ndim == 3:
        # Otherwise also reverse channels dimension
        return np.moveaxis(obs, 0, 1)[:, :, ::-1]
    else:
        rgb_obs = np.repeat(obs[:, :, None], 3, axis=2)
        return np.moveaxis(rgb_obs, 0, 1)


def play_env(env, key_map, real_time=False, bgra=False):
    obs, _ = env.reset()

    window = Window(image=obs, bgra=bgra)

    keys = window.update(obs)

    # keys = set()
    terminal = False
    while 'q' not in keys:
        # Only step the environment if we have keys pressed or we're running in
        # real time.
        if len(keys) > 0:
            for key_name, num in key_map.items():
                if key_name in keys:
                    action = num

            obs, reward, terminal, _, _ = env.step(action)
            keys = window.update(obs)
            print(reward)
        elif real_time:
            action = key_map['nothing']
            obs, reward, terminal, _, _ = env.step(action)
            keys = window.update(obs)
            print(reward)

        if terminal:
            print('terminated')
            sleep(1)
            obs, _ = env.reset()
            keys = window.update(obs)

        sleep(0.017)

    window.close()


def play_breakout():
    env = gym.make('ALE/Breakout-v5')

    key_map = {'left': 3, 'up': 1, 'right': 2, 'down': 0, 'nothing': 0}

    play_env(env, key_map, real_time=True)


def display_observations(observations):
    window = Window(image=observations[0])

    for obs in observations:
        keys = window.update(obs)

        if 'q' in keys:
            break


def sdl_io_loop(in_q, out_q, image, name, bgra):
    sdl2.ext.init()

    match image.shape:
        case h, w:
            height = h
            width = w
        case h, w, _:
            height = h
            width = w

    window = sdl2.ext.Window(name, size=(width, height))

    view = pixels3d(window.get_surface())

    view[:, :, :3] = reformat_image(image, bgra)

    window.show()

    keys = set()
    running = True
    while running:
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                out_q.put('q')
                break
            elif event.type == sdl2.SDL_KEYDOWN:
                match event.key.keysym.sym:
                    case sdl2.SDLK_q:
                        running = False
                        # out_q.put('q')
                        keys.add('q')
                        out_q.put(keys)
                        break
                    case sdl2.SDLK_LEFT:
                        # out_q.put('left')
                        keys.add('left')
                    case sdl2.SDLK_UP:
                        # out_q.put('up')
                        keys.add('up')
                    case sdl2.SDLK_RIGHT:
                        # out_q.put('right')
                        keys.add('right')
                    case sdl2.SDLK_DOWN:
                        # out_q.put('down')
                        keys.add('down')
                    case sdl2.SDLK_RETURN:
                        # out_q.put('enter')
                        keys.add('enter')
            elif event.type == sdl2.SDL_KEYUP:
                match event.key.keysym.sym:
                    case sdl2.SDLK_LEFT:
                        keys.remove('left')
                    case sdl2.SDLK_UP:
                        keys.remove('up')
                    case sdl2.SDLK_RIGHT:
                        keys.remove('right')
                    case sdl2.SDLK_DOWN:
                        keys.remove('down')
                    case sdl2.SDLK_RETURN:
                        keys.remove('enter')

        # Respond to a new image after getting key events in order to return
        # slightly less out-of-date key events.
        if not in_q.empty():
            image = in_q.get()

            if image is None:
                break

            view[:, :, :3] = reformat_image(image, bgra)
            out_q.put(keys)

        window.refresh()

        sleep(0.01)

    window.close()
    sdl2.ext.quit()
    # Possibly not necessary
    # in_q.cancel_join_thread()
    # out_q.cancel_join_thread()


def test_window():
    from common import rng

    image = np.uint8((rng.random(250_000) * 255)).reshape((500, 500))

    window = Window(image)

    keys = set()
    while 'q' not in keys:
        image = np.uint8((rng.random(250_000) * 255)).reshape((500, 500))
        keys = window.update(image)
        print('Keys being pressed:', ' '.join(keys))
        sleep(1)

    window.close()


def plot_loop(in_q, out_q):
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")

    plt.ion()

    figure, axes = plt.subplots(figsize=(10, 5), layout='constrained')
    dr_line, = axes.plot([], [], label='discounted reward')
    mq_line, = axes.plot([], [], label='max Q-value')
    axes.legend(fontsize=15, loc='upper left')
    axes.set_ylim(-0.1, 2.0)

    while True:
        d_rewards, step_counts, q_values = in_q.get()

        # print(list(zip(np.arange(d_rewards.shape[0]), d_rewards)))

        axes.set_xlim(0, d_rewards.shape[0])

        dr_line.set_data(np.arange(d_rewards.shape[0]), d_rewards)
        mq_line.set_data(step_counts, q_values)

        # axes.relim()
        # axes.autoscale_view()
        figure.canvas.draw_idle()
        figure.canvas.flush_events()


def test_plotter():
    from common import rng

    plotter = Plotter()

    while True:
        test_data = (
            rng.random(100) * 2,
            np.arange(0, 100, 2),
            rng.random(50)
        )
        plotter.put(test_data)
        sleep(1)


class Plotter(Worker):
    def __init__(self):
        super().__init__(plot_loop)


class Window(Worker):
    def __init__(self, image, name='SDL', bgra=False):
        super().__init__(sdl_io_loop, image, name, bgra)

    def update(self, image):
        self.put(image)
        return self.get()


if __name__ == '__main__':
    play_breakout()
