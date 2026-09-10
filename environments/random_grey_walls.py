from random import random

import jax
import jax.numpy as jnp
import jax.random as jrd


@jax.jit(static_argnums=0)
def get_observation(size, gap_indices, out_of_bounds, y, x):
    obs = jnp.zeros((size, size), dtype=jnp.uint8)

    wall_indices = jnp.arange(1, size, 2)

    obs = obs.at[wall_indices].set(128)
    obs = obs.at[wall_indices, gap_indices].set(0)

    obs = obs.at[y, x].set(jnp.where(jnp.logical_not(out_of_bounds), 255, 0))

    return obs


@jax.jit(static_argnums=1)
def step(action, size, gap_indices, safe, y, x):
    # Record this before applying the action.
    bottom_right = jnp.logical_and(x == size - 1, y == size - 1)

    # left up right down
    x = jnp.where(action == 0, x - 1, x)
    y = jnp.where(action == 1, y - 1, y)
    x = jnp.where(action == 2, x + 1, x)
    y = jnp.where(action == 3, y + 1, y)

    in_room_x = jnp.logical_and(x >= 0, x < size)
    in_room_y = jnp.logical_and(y >= 0, y < size)

    in_room = jnp.logical_and(in_room_x, in_room_y)

    # The same as using jnp.floor. Also jnp.int32(-0.5) == 0.
    gap_y = jnp.int32(y / 2)

    gap_x = jnp.where(gap_y < len(gap_indices), gap_indices[gap_y], 0)

    aligned_with_gap = x == gap_x

    on_even_row = y % 2 == 1

    on_wall_row = jnp.logical_and(in_room, on_even_row)

    in_wall = jnp.logical_and(on_wall_row, jnp.logical_not(aligned_with_gap))

    x = jnp.where(safe, jnp.clip(x, 0, size - 1), x)
    y = jnp.where(safe, jnp.clip(y, 0, size - 1), y)

    # Safe boundaries stop the player from going out of bounds.
    undo_action = jnp.logical_and(safe, in_wall)

    x = jnp.where(jnp.logical_and(undo_action, action == 0), x + 1, x)
    y = jnp.where(jnp.logical_and(undo_action, action == 1), y + 1, y)
    x = jnp.where(jnp.logical_and(undo_action, action == 2), x - 1, x)
    y = jnp.where(jnp.logical_and(undo_action, action == 3), y - 1, y)

    in_wall = jnp.where(undo_action, False, in_wall)

    out_of_x_bounds = jnp.logical_or(x < 0, x >= size)
    out_of_y_bounds = jnp.logical_or(y < 0, y >= size)

    out_of_x_or_y_bounds = jnp.logical_or(out_of_x_bounds, out_of_y_bounds)

    out_of_bounds = jnp.logical_or(out_of_x_or_y_bounds, in_wall)

    # To get the reward the agent has to move rightwards from the bottom right
    # corner of the maze.
    reward_found = jnp.logical_and(bottom_right, action == 2)

    reward = jnp.where(reward_found, 1.0, 0.0)

    terminated = jnp.logical_or(out_of_bounds, reward_found)

    obs = get_observation(size, gap_indices, out_of_bounds, y, x)

    bottom_right_pixel = obs[size - 1, size - 1]

    # If we were blocked from exiting by safe_boundaries then set the bottom
    # right pixel to zero, to indicate that we have exited. Has no effect on
    # training.
    obs = obs.at[size - 1, size - 1].set(
        jnp.where(reward_found, 0, bottom_right_pixel)
    )

    info_y_clamped = jnp.clip(y, 0, size)
    info_y_clamped = jnp.where(action == 1, y + 1, y)
    info_y_clamped = jnp.where(action == 3, y - 1, y)

    info_y = jnp.where(terminated, info_y_clamped, y)

    info_x_clamped = jnp.clip(x, 0, size)
    info_x_clamped = jnp.where(action == 0, x + 1, x)
    info_x_clamped = jnp.where(action == 2, x - 1, x)

    info_x = jnp.where(terminated, info_x_clamped, x)

    info = {'y': info_y, 'x': info_x}

    return obs, reward, terminated, info, y, x


class Env:
    def __init__(self, size=7, safe_boundaries=False):
        self.size = size
        self.safe_boundaries = safe_boundaries

        self.key = jrd.key(int(random() * 1e12))

        self.obs_size = jnp.int32(size)
        self.num_actions = jnp.uint8(4)

        self.eval_data = None

    def reset(self):
        self.gap_indices = jrd.choice(
            self.key, self.size, shape=(int(self.size / 2),)
        )

        if self.size % 2 == 0:
            self.gap_indices = self.gap_indices.at[-1].set(self.size - 1)

        self.key, self.subkey = jrd.split(self.key)

        self.player_x = jnp.int32(0)
        self.player_y = jnp.int32(0)

        self.out_of_bounds = jnp.bool(False)

        return get_observation(
            self.size, self.gap_indices, self.out_of_bounds,
            self.player_y, self.player_x
        )

    def step(self, action):
        obs, reward, terminated, info, self.player_y, self.player_x = step(
            action, self.size, self.gap_indices, self.safe_boundaries,
            self.player_y, self.player_x
        )

        return obs, reward, terminated, False, info

    def close(self):
        pass

    def get_eval_data(self):
        observations = []
        rows = []
        cols = []

        prev_row = 0
        prev_col = 0

        for row in range(self.size):
            for col in range(self.size):
                if row % 2 == 0 or col == self.gap_indices[int(row / 2)]:
                    self.observation[prev_row, prev_col] = 0
                    self.observation[row, col] = 255
                    rows.append(row)
                    cols.append(col)

                    observations.append(self.observation.copy())

                    prev_row = row
                    prev_col = col

        self.observation[prev_row, prev_col] = 0

        self.observation[0, 0] = 255

        return np.array(observations), rows, cols

    def get_renderable_q_values(self, net):
        if self.eval_data is None:
            self.eval_data = self.get_eval_data()

        observations, rows, cols = self.eval_data

        q_values = net(observations[:, :, None])

        max_qs = np.array(np.max(q_values, axis=1))

        negative_max_qs = np.clip(max_qs, a_min=-1.0, a_max=0.0)

        positive_max_qs = np.clip(max_qs, a_min=0.0, a_max=1.0)

        negative_converted = np.array(negative_max_qs * 255, dtype=np.uint8)
        positive_converted = np.array(positive_max_qs * 255, dtype=np.uint8)

        background = observations[0].copy()
        background[0, 0] = 0
        background = np.repeat(background[:, :, None], 3, axis=2)

        # Set green & blue channels to zero
        background[:, :, 1:] = 0

        # Add a dimension for broadcasting
        background[rows, cols, :] = positive_converted[:, None]

        background[rows, cols, 0] -= negative_converted

        return background

        reshaped = positive_max_qs.reshape((self.obs_size, self.obs_size))

        repeated_c = np.repeat(reshaped[:, :, None], 3, axis=2)

        # Decrement rather than assign so we don't change the red channel for
        # the positive Q-values.
        repeated_c[:, :, 0] -= (
            negative_max_qs.reshape((self.obs_size, self.obs_size))
        )

        converted = np.array(repeated_c * 255, dtype=np.uint8)
        return converted


if __name__ == '__main__':
    env = Env(size=2, safe_boundaries=True)

    obs = env.reset()
    print('Enter \'q\' to quit')
    while True:
        print(obs)

        try:
            action = int(input())
        except ValueError:
            break

        obs_next, reward, term, _, _ = env.step(action)

        print(reward, term, '\n')

        obs = obs_next

        if term:
            print(obs)
            print('Resetting')
            obs = env.reset()

    env.reset()

    import numpy as np

    rng = np.random.default_rng()

    step_counts = []
    returns = []
    reward_count = 0

    num_episodes = 1_000
    max_steps = 10_000

    for _ in range(num_episodes):
        env.reset()
        for step_count in range(max_steps):
            action = rng.choice(env.num_actions)
            obs, reward, term, _, _ = env.step(action)

            assert reward == 0 or reward == 1

            if term:
                print(f'Found terminal state after {step_count + 1} actions')

                step_counts.append(step_count + 1)
                returns.append(reward)
                reward_count += int(bool(reward))
                break

            if step_count == max_steps - 1:
                print('Truncating')
                step_counts.append(step_count + 1)
                returns.append(reward)

    print(f'Average return: {np.mean(np.array(returns)):.2f}')
    print(f'Average steps: {int(np.mean(np.array(step_counts)))}')

    percentage = round(100 * reward_count / num_episodes)
    print(f'Found reward state in {percentage}% of episodes')
