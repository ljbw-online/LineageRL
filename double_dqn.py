from random import random, randint

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrd
from flax import nnx
import optax


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

    # while_loop :: (a -> Bool) -> (a -> a) -> (a -> a)
    replay_buffer = lax.while_loop(
        overlap_check,
        update_least_recent_bounds,
        replay_buffer
    )

    return replay_buffer


def add_timestep_to_episode(timestep, episode):
    step_count = episode['step_count']

    episode['observation'] = (
        episode['observation'].at[step_count].set(timestep['observation'])
    )

    # Along with each observation we get the action, reward and terminated
    # value for the previous step. When step_count == 0 we just write
    # placeholders, and then write the actual values on the next step.
    prev_step_count = jnp.maximum(0, step_count - 1)

    for key in ['action', 'reward', 'terminated']:
        episode[key] = episode[key].at[prev_step_count].set(timestep[key])

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
    episode = add_timestep_to_episode(timestep, episode)

    # We reached the max episode length or we got terminated on the previous
    # step.
    should_add_episode = jnp.logical_or(
        episode['step_count'] == replay_buffer['max_episode_length'],
        terminated
    )

    replay_buffer = lax.cond(
        should_add_episode, add_episode, lambda rb: rb,
        replay_buffer
    )

    # Reset step_count if we added the episode
    replay_buffer['episode']['step_count'] = (
        jnp.where(should_add_episode, 0, episode['step_count'])
    )

    return replay_buffer


@jax.jit
def generate_batch(state):
    # batch_size = 32
    replay_buffer = state['replay_buffer']
    batch = replay_buffer['batch']
    sequence_length = replay_buffer['batch']['action'].shape[1]

    add_bounds_index = replay_buffer['add_bounds_index']
    least_recent_bounds_index = replay_buffer['least_recent_bounds_index']
    max_episodes = replay_buffer['boundaries'].shape[0]

    # Using the mod operator makes this correct even when
    # add_bounds_index < least_recent_bounds_index.
    # This isn't correct if both of them are zero, but that isn't possible
    # when the number of bounds is equal to the buffer length and the
    # shortest episode is two timesteps.
    num_stored_episodes = (
        (add_bounds_index - least_recent_bounds_index) % max_episodes
    )

    state['key'], subkey = jrd.split(state['key'])

    # TODO: sample episodes with probability proportional to their length!

    # Episode choices, using randint because jrd.choice doesn't like tracers.
    choices = (
        (jrd.randint(subkey, (32,), 0, num_stored_episodes)
            + least_recent_bounds_index)
        % max_episodes
    )

    # observation_shape = replay_buffer['observation'].shape[1:]

    # obs_batch_shape = (
    #     (batch_size,) + observation_shape + (sequence_length,)
    # )

    choice_bounds = replay_buffer['boundaries'][choices]

    # batch = {
    #     'observation': jnp.zeros(obs_batch_shape, dtype=jnp.uint8),
    #     'action': jnp.zeros((batch_size, sequence_length), dtype=jnp.int32),
    #     'reward': jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
    #     'terminated': jnp.zeros((batch_size, sequence_length), dtype=jnp.bool)
    # }

    for i, bounds in enumerate(choice_bounds):
        lower = bounds[0]
        upper = bounds[1]

        ep_len = upper - lower

        # TODO: ensure sequences ending before t=seq_len get into batches

        state['key'], subkey = jrd.split(state['key'])
        # seq_start = jrd.choice(subkey, ep_len - sequence_length + 1)
        seq_start = jrd.randint(subkey, (), 0, ep_len - sequence_length + 1)
        # seq_end = seq_start + sequence_length

        seq_slice = jax.ds(seq_start, sequence_length)

        # Return observations channels-last
        batch['observation'] = batch['observation'].at[i].set(
            jnp.moveaxis(
                # replay_buffer['observation'][seq_start: seq_end], 0, -1
                replay_buffer['observation'][seq_slice], 0, -1
            )
        )

        for k in ['action', 'reward', 'terminated']:
            # batch[k] = batch[k].at[i].set(replay_buffer[k][seq_start: seq_end])
            batch[k] = batch[k].at[i].set(replay_buffer[k][seq_slice])

    state['batch'] = batch

    return state


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


def display_batch(batch):
    for i in range(batch['action'].shape[0]):
        frame0 = batch['observation'][i, :, :, 0]
        frame1 = batch['observation'][i, :, :, 1]

        print(frame0)
        print(frame1)

        print(batch['action'][i], batch['reward'][i], batch['terminated'][i])
        print()


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


def get_regression_targets(state):
    batch = state['batch']
    main_net = state['main_net']
    target_net = state['target_net']

    next_step_obs = batch['observation'][:, :, :, 1:]

    main_next_step_action_values = main_net(next_step_obs)
    target_next_step_action_values = target_net(next_step_obs)

    main_next_step_selected_actions = jnp.argmax(
        main_next_step_action_values, axis=1
    )

    target_values_of_main_actions = target_next_step_action_values[
        jnp.arange(next_step_obs.shape[0]), main_next_step_selected_actions
    ]

    rewards = batch['reward'][:, 0]
    terminated = batch['terminated'][:, 0]
    gamma = state['gamma']

    targets = (
        rewards
        + jnp.logical_not(terminated) * gamma * target_values_of_main_actions
    )

    return targets


def loss_fn(main_net, loss_batch):
    main_q_values = main_net(loss_batch['observation'])
    actions = loss_batch['action']
    targets = loss_batch['target']

    selected_q_values = main_q_values[
        jnp.arange(main_q_values.shape[0]), actions[:, 0]
    ]

    return jnp.mean((selected_q_values - targets) ** 2)


loss_with_gradients = nnx.value_and_grad(loss_fn)


def update_target_net(state):
    main_net = state['main_net']
    target_net = state['target_net']

    _, main_params = nnx.split(main_net, nnx.Param)
    # TODO: does this in-place update get into the state dict?
    nnx.update(target_net, main_params)

    state['main_updates_since_target_update'] = 0

    return state


def update_nets(state):
    state = generate_batch(state)

    main_net = state['main_net']
    # seq_len = state['seq_len']
    optimiser = state['optimiser']

    batch = state['batch']
    targets = get_regression_targets(state)

    loss_batch = {
        'observation': batch['observation'][:, :, :, :1],
        'action': batch['action'],
        'target': targets
    }

    loss, grads = loss_with_gradients(main_net, loss_batch)

    # TODO: does this in-place update get into the state dict?
    optimiser.update(main_net, grads)

    state['main_updates_since_target_update'] += 1
    state['loss'] = loss

    should_update_target = (
        state['main_updates_since_target_update']
        >= state['main_updates_per_target_update']
    )

    state = lax.cond(
        should_update_target,
        update_target_net,
        lambda s: s,
        state
    )

    return state


@jax.jit
def train(timestep, state):
    state['key'], subkey = jrd.split(state['key'])
    # TODO: epsilon-greedy
    action = jrd.randint(subkey, (), 0, state['num_actions'])

    rb = state['replay_buffer']
    state['replay_buffer'] = add_timestep(timestep, rb)

    state = update_nets(state)

    return action, state


def get_replay_buffer(
    buffer_length, max_episode_length, observation_shape, sequence_length
):
    observation_buffer_shape = (buffer_length,) + observation_shape
    observation_episode_shape = (max_episode_length,) + observation_shape
    return {
        'observation': jnp.zeros(observation_buffer_shape, dtype=jnp.uint8),
        'action': jnp.zeros(buffer_length, dtype=jnp.int32),
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
            'action': jnp.zeros(max_episode_length, dtype=jnp.int32),
            'reward': jnp.zeros(max_episode_length, dtype=jnp.float32),
            'terminated': jnp.zeros(max_episode_length, dtype=jnp.bool),
            'step_count': jnp.int32(0),
        },
        'batch': {
            'observation': jnp.zeros(
                (32,) + observation_shape + (sequence_length,), dtype=jnp.uint8
            ),
            'action': jnp.zeros((32, 2), dtype=jnp.int32),
            'reward': jnp.zeros((32, 2), dtype=jnp.float32),
            'terminated': jnp.zeros((32, 2), dtype=jnp.bool)
        }
    }


def get_state(
    buffer_length, max_episode_length, obs_shape, num_actions, seq_len
):
    rngs = nnx.Rngs(int(random() * 1e12))
    main_net = Qnet(obs_shape, num_actions, rngs)
    target_net = Qnet(obs_shape, num_actions, rngs)
    optimiser = nnx.Optimizer(main_net, optax.adam(1e-4), wrt=nnx.Param)
    return {
        'key': jrd.key(int(random() * 1e12)),
        'replay_buffer': get_replay_buffer(
            buffer_length, max_episode_length, obs_shape, seq_len
        ),
        'num_actions': num_actions,
        'main_net': main_net,
        'target_net': target_net,
        'optimiser': optimiser,
        'seq_len': 2,
        'gamma': 0.99,
        'num_steps': 0,
        'main_updates_since_target_update': 0
    }


class Qnet(nnx.Module):
    def __init__(self, obs_shape, num_actions, rngs):
        self.size = obs_shape[0] * obs_shape[1]
        self.linear0 = nnx.Linear(self.size, 128, rngs=rngs)
        self.linear1 = nnx.Linear(128, num_actions, rngs=rngs)

    def __call__(self, obs_batch):
        obs_batch = obs_batch.reshape(obs_batch.shape[0], -1) / 255.0
        a = self.linear0(obs_batch)
        a = nnx.gelu(a)
        a = self.linear1(a)
        return a


class Agent:
    def __init__(
        self, env_constructor, buffer_len=1e6, max_ep_len=18e3, seq_len=2
    ):
        self.max_ep_len = max_ep_len

        env = env_constructor()
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n

        self.env = env

        # self.state = get_state(
        #     buffer_len, max_ep_len, obs_shape, num_actions, seq_len
        # )

        rngs = nnx.Rngs(int(random() * 1e12))
        main_net = Qnet(obs_shape, num_actions, rngs)
        target_net = Qnet(obs_shape, num_actions, rngs)
        optimiser = nnx.Optimizer(main_net, optax.adam(1e-4), wrt=nnx.Param)

        self.state = {
            'key': jrd.key(int(random() * 1e12)),
            'replay_buffer': get_replay_buffer(
                buffer_len, max_ep_len, obs_shape, seq_len
            ),
            'num_actions': num_actions,
            'main_net': main_net,
            'target_net': target_net,
            'optimiser': optimiser,
            # 'seq_len': 2,
            'gamma': 0.99,
            # 'num_steps': 0,
            'main_updates_since_target_update': 0,
            'main_updates_per_target_update': 100
        }

    def train(self, num_steps):
        step_count = 0
        while True:
            obs, _ = self.env.reset()
            terminated = False
            timestep = {
                'observation': obs,
                'action': 0,
                'reward': 0,
                'terminated': False
            }
            ep_return = 0
            for i in range(self.max_ep_len):
                step_count += 1

                action, self.state = train(timestep, self.state)

                mustu = self.state['main_updates_since_target_update']
                if mustu == 99 or mustu == 0:
                    print('mustu:', mustu)
                    print(self.state['main_net'](obs[None, :]))
                    print(self.state['target_net'](obs[None, :]))

                if terminated:
                    break

                obs_next, reward, terminated, _, _ = self.env.step(action)
                timestep = {
                    'observation': obs_next,
                    'action': action,
                    'reward': reward,
                    'terminated': terminated
                }
                ep_return += reward

            # print(f'length {i + 1}, return {ep_return}')

            if step_count >= num_steps:
                break


if __name__ == '__main__':
    from environments.follow_the_dots import Env

    def get_env():
        return Env(2)

    agent = Agent(get_env, buffer_len=1_000, max_ep_len=20)

    agent.train(250)

    rb = agent.state['replay_buffer']

    # assert any(rb['terminated'])

    # while True:
    #     batch = get_batch(agent.state)
    #     display_batch(batch)
        # input('Press Enter\n')
