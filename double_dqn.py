from random import random, randint

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrd


def circular_get(buffer, start):
    return buffer[start], (start + 1) % buffer.shape[0]


def circular_put(buffer, end, item):
    return buffer.at[end].set(item), (end + 1) % buffer.shape[0]


def write_timestep(i, data):
    replay_buffer, buffer_offset = data

    for key in ['observation', 'action', 'reward', 'terminated']:
        replay_buffer[key] = replay_buffer[key].at[buffer_offset + i].set(
            replay_buffer['episode'][key][i]
        )

    # jax.debug.print(
    #     'i={x}, act={y}', x=i,
    #     y=replay_buffer['episode']['action'][i]
    # )

    # We have to return buffer_offset because a loop body has to return
    # something with the same pytree structure as what it receives.
    return replay_buffer, buffer_offset


def add_episode(replay_buffer):
    buffer_length = replay_buffer['observation'].shape[0]

    length = replay_buffer['episode']['step_count']

    boundaries = replay_buffer['boundaries']
    add_bounds_index = replay_buffer['add_bounds_index']
    start_index = replay_buffer['add_episode_index']

    buffer_end_reached = start_index + length > buffer_length

    # If there isn't enough space for the episode at the end of the buffer
    # then put it at the beginning, otherwise put it directly after the
    # previous episode.
    start_index = jnp.where(
        buffer_end_reached,
        0,
        start_index
    )

    end_index = start_index + length

    boundaries, add_bounds_index = circular_put(
        boundaries, add_bounds_index, [start_index, end_index]
    )

    replay_buffer['boundaries'] = boundaries
    replay_buffer['add_bounds_index'] = add_bounds_index
    replay_buffer['add_episode_index'] = end_index

    replay_buffer, _ = lax.fori_loop(
        0, length, write_timestep,
        (replay_buffer, start_index)
    )

    # Record whether the buffer capacity has been reached so far
    replay_buffer['wrapped'] = jnp.logical_or(
        replay_buffer['wrapped'], buffer_end_reached
    )

    def overlap_check(replay_buffer):
        # jax.debug.print(
        #     'cap_r {a}, end_i {b}, bound {c}', a=capacity_reached,
        #     b=end_index, c=boundaries[first_ep_pos, 0]
        # )
        least_recent_ep_start = replay_buffer['least_recent_bounds'][0]

        return jnp.logical_and(
            replay_buffer['wrapped'],
            end_index < least_recent_ep_start
        )

    def update_least_recent_bounds(replay_buffer):
        boundaries = replay_buffer['boundaries']
        least_recent_bounds_index = replay_buffer['least_recent_bounds_index']

        least_recent_bounds, least_recent_bounds_index = circular_get(
            boundaries, least_recent_bounds_index
        )

        replay_buffer['least_recent_bounds'] = least_recent_bounds
        replay_buffer['least_recent_bounds_index'] = least_recent_bounds_index

        return replay_buffer

    # jax.debug.print('before while')
    # while_loop :: (a -> Bool) -> (a -> a) -> (a -> a)
    replay_buffer = lax.while_loop(
        overlap_check,
        update_least_recent_bounds,
        replay_buffer
    )
    # jax.debug.print('after while')

    return replay_buffer


def add_timestep_to_episode(
    timestep, episode
):
    step_count = episode['step_count']

    for key in ['observation', 'action', 'reward', 'terminated']:
        # if key == 'observation':
        #     jax.debug.print('obs shape {x}', x=episode[key][step_count].shape)
        #     jax.debug.print('ts shape {x}', x=timestep[key].shape)
        episode[key] = episode[key].at[step_count].set(timestep[key])

    episode['step_count'] = step_count + 1

    return episode


@jax.jit
def add_timestep(timestep, replay_buffer):
    # Record whether we got terminated on the previous step
    episode = replay_buffer['episode']
    # Jax clamps too-large indices but wraps negative indices
    prev_term_index = jnp.maximum(0, episode['step_count'] - 1)
    terminated = episode['terminated'][prev_term_index]

    # Increments step_count
    episode = (
        add_timestep_to_episode(timestep, episode)
    )

    # We reached the max episode length or we got terminated on the previous
    # step.
    should_add_episode = jnp.logical_or(
        episode['step_count'] == replay_buffer['max_episode_length'],
        terminated
    )

    # lax.cond(
    #     should_add_episode,
    #     lambda pair: jax.debug.print(
    #         's_c {a}, mel {b}, s_c==mel {c}, prev_term {d}',
    #         a=episode['step_count'],
    #         b=replay_buffer['max_episode_length'],
    #         c=episode['step_count'] == replay_buffer['max_episode_length'],
    #         d=terminated
    #     ),
    #     lambda _: None,
    #     (timestep['action'], episode['step_count'])
    # )

    # jax.debug.print('before add_ep')
    replay_buffer = lax.cond(
        should_add_episode, add_episode, lambda rb: rb,
        replay_buffer
    )
    # jax.debug.print('after add_ep')

    # Reset step_count if we added the episode
    replay_buffer['episode']['step_count'] = (
        jnp.where(should_add_episode, 0, episode['step_count'])
    )

    return replay_buffer


@jax.jit(static_argnums=2)
def get_batch(key, replay_buffer, sequence_length):
    batch_size = 32
    add_bounds_index = replay_buffer['add_bounds_index']
    least_recent_bounds_index = replay_buffer['least_recent_bounds_index']
    max_episodes = replay_buffer['boundaries'].shape[0]
    # buffer_length = replay_buffer['action'].shape[0]

    # The mod operator keeps this positive
    offset_range = (
        (add_bounds_index - least_recent_bounds_index) % max_episodes
    )

    choice_range = max_episodes

    key, subkey = jrd.split(key)

    # Episode choices
    choices = (
        (jrd.choice(subkey, choice_range, shape=(32,)) + add_bounds_index)
        % max_episodes
    )

    observation_shape = replay_buffer['observation'].shape[1:]

    obs_batch_shape = (
        (batch_size,) + observation_shape + (sequence_length,)
    )

    choice_bounds = replay_buffer['boundaries'][choices]

    batch = {
        'observation': jnp.zeros(obs_batch_shape, dtype=jnp.uint8),
        'action': jnp.zeros((batch_size, sequence_length), dtype=jnp.uint8),
        'reward': jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
        'terminated': jnp.zeros((batch_size, sequence_length), dtype=jnp.bool)
    }

    for i, bounds in enumerate(choice_bounds):
        lower = bounds[0]
        upper = bounds[1]

        # Unnecessary because we're not storing episodes across the buffer
        # boundary.
        # ep_len = jnp.where(
        #     upper > lower,
        #     upper - lower,
        #     max_episodes - lower + upper
        # )

        ep_len = upper - lower

        key, subkey = jrd.split(key)
        seq_start = jrd.choice(subkey, ep_len - sequence_length + 1)
        seq_end = seq_start + sequence_length

        # Also unnecessary
        # start_index = (lower + seq_start) % buffer_length
        # end_index = (start_index + sequence_length) % buffer_length

        # Return observations channels-last
        batch['observation'] = batch['observation'].at[i].set(
            jnp.moveaxis(
                replay_buffer['observation'][seq_start: seq_end], 0, -1
            )
        )

        for k in ['action', 'reward', 'terminated']:
            batch[k] = batch[k].at[i].set(replay_buffer[k][seq_start: seq_end])

    return key, batch


def get_replay_buffer(
    buffer_length, max_episode_length, observation_shape
):
    observation_buffer_shape = (buffer_length,) + observation_shape
    observation_episode_shape = (max_episode_length,) + observation_shape
    return {
        'observation': jnp.zeros(observation_buffer_shape, dtype=jnp.uint8),
        'action': jnp.zeros(buffer_length, dtype=jnp.uint8),
        'reward': jnp.zeros(buffer_length, dtype=jnp.float32),
        'terminated': jnp.zeros(buffer_length, dtype=jnp.bool),
        'boundaries': jnp.zeros((buffer_length, 2), dtype=jnp.int32),
        'add_bounds_index': jnp.int32(0),
        'least_recent_bounds_index': jnp.int32(0),
        'least_recent_bounds': jnp.zeros(2, dtype=jnp.int32),
        'add_episode_index': jnp.int32(0),
        'wrapped': jnp.bool(0),
        'max_episode_length': jnp.int32(max_episode_length),
        'episode': {
            'observation': jnp.zeros(
                observation_episode_shape, dtype=jnp.uint8
            ),
            'action': jnp.zeros(max_episode_length, dtype=jnp.uint8),
            'reward': jnp.zeros(max_episode_length, dtype=jnp.float32),
            'terminated': jnp.zeros(max_episode_length, dtype=jnp.bool),
            'step_count': jnp.int32(0),
        }
    }


def add_test_episode(replay_buffer, max_episode_length, j):
    if j > max_episode_length:
        raise ValueError

    for i in range(1, j + 1):
        if j == max_episode_length:
            # Truncated episode
            term = i == j - 1 and random() < 0.5
        else:
            term = i == j - 1

        ts = {
            'observation': jnp.uint8(j ** i),
            'action': jnp.uint8(j ** i),
            'reward': jnp.float32(j ** i),
            'terminated': jnp.bool(term),
        }

        replay_buffer = add_timestep(ts, replay_buffer)

    return replay_buffer


def add_random_test_episode(replay_buffer, max_episode_length):
    # Randint is inclusive of the upper bound
    j = randint(2, max_episode_length)

    return add_test_episode(replay_buffer, max_episode_length, j)


def run_episode(env, replay_buffer):
    max_len = replay_buffer['max_episode_length']
    observation = env.reset()
    terminated = False

    for _ in range(max_len):
        if terminated:
            replay_buffer = add_timestep(
                {
                    'observation': observation,
                    'action': 5,
                    'reward': 0,
                    'terminated': False
                },
                replay_buffer
            )
            break

        action = randint(1, 3)
        observation_next, reward, terminated, _, _ = env.step(action)

        timestep = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'terminated': terminated
        }

        replay_buffer = add_timestep(timestep, replay_buffer)

        observation = observation_next

    return replay_buffer


class Agent:
    def __init__(self):
        pass


if __name__ == '__main__':
    from time import sleep
    from environments.random_grey_walls import Env
    max_ep_len = 10
    env = Env(size=2, safe_boundaries=True)

    rb = get_replay_buffer(30, max_ep_len, (2, 2))

    key = jrd.key(int(random() * 1e9))

    for i in range(300):
        rb = run_episode(env, rb)

        sleep(1)

        print(rb['action'])
        print(get_batch(key, rb, 2))
        input()
