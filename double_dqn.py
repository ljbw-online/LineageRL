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

    # We have to return buffer_offset because a loop body has to return
    # something with the same pytree structure as what it receives.
    return replay_buffer, buffer_offset


def add_episode(replay_buffer):
    buffer_length = replay_buffer['observation'].shape[0]

    length = replay_buffer['episode']['step_count']

    replay_buffer['timesteps_stored'] += length

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
        least_recent_bounds, least_recent_bounds_index = circular_get(
            replay_buffer['boundaries'],
            replay_buffer['least_recent_bounds_index']
        )

        start = least_recent_bounds[0]
        end = least_recent_bounds[1]

        replay_buffer['timesteps_stored'] -= end - start

        replay_buffer['least_recent_bounds'] = least_recent_bounds
        replay_buffer['least_recent_bounds_index'] = least_recent_bounds_index

        return replay_buffer

    # while_loop :: (a -> Bool) -> (a -> a) -> (a -> a)
    replay_buffer = lax.while_loop(
        overlap_check,
        update_least_recent_bounds,
        replay_buffer
    )

    add_bounds_index = replay_buffer['add_bounds_index']
    least_recent_bounds_index = replay_buffer['least_recent_bounds_index']
    timesteps_stored = replay_buffer['timesteps_stored']
    probs = replay_buffer['episode_probabilities']

    bounds_wrapped = add_bounds_index < least_recent_bounds_index

    for i, bounds in enumerate(boundaries):
        start = bounds[0]
        end = bounds[1]

        # Is the current index inside or outside the circular segment
        # containing the active episode boundaries.
        inside = jnp.logical_and(
            i > least_recent_bounds_index, i < add_bounds_index
        )

        outside = jnp.logical_or(
            i < add_bounds_index, i > least_recent_bounds_index
        )

        # inside = i > least_recent_bounds_index and i < add_bounds_index
        # outside = i < add_bounds_index or i > least_recent_bounds_index

        set_prob = jnp.logical_or(
            jnp.logical_and(jnp.logical_not(bounds_wrapped), inside),
            jnp.logical_and(bounds_wrapped, outside)
        )

        # set_prob = (
        #     (not bounds_wrapped and inside)
        #     or (bounds_wrapped and outside)
        # )

        prob = jnp.where(set_prob, (end - start) / timesteps_stored, 0)

        probs = probs.at[i].set(prob)

    replay_buffer['episode_probabilities'] = probs

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
    replay_buffer = state['replay_buffer']
    batch = replay_buffer['batch']
    batch_size = replay_buffer['batch']['action'].shape[0]
    sequence_length = replay_buffer['batch']['action'].shape[1]

    # add_bounds_index = replay_buffer['add_bounds_index']
    # least_recent_bounds_index = replay_buffer['least_recent_bounds_index']
    # max_episodes = replay_buffer['boundaries'].shape[0]

    # Using the mod operator makes this correct even when
    # add_bounds_index < least_recent_bounds_index.
    # This isn't correct if both of them are zero, but that isn't possible
    # when the number of bounds is equal to the buffer length and the
    # shortest episode is two timesteps.
    # num_stored_episodes = (
    #     (add_bounds_index - least_recent_bounds_index) % max_episodes
    # )

    state['key'], subkey = jrd.split(state['key'])

    # TODO: sample episodes with probability proportional to their length!

    # Episode choices, using randint because jrd.choice doesn't like tracers.
    # choices = (
    #     (jrd.randint(subkey, (32,), 0, num_stored_episodes)
    #         + least_recent_bounds_index)
    #     % max_episodes
    # )

    # choice_bounds = replay_buffer['boundaries'][choices]

    boundaries = replay_buffer['boundaries']
    sample_probs = replay_buffer['episode_probabilities']

    choice_bounds = jrd.choice(
        subkey, boundaries, shape=(batch_size,), p=sample_probs
    )

    for i, bounds in enumerate(choice_bounds):
        lower = bounds[0]
        upper = bounds[1]

        ep_len = upper - lower

        # TODO: ensure sequences ending before t=seq_len get into batches

        state['key'], subkey = jrd.split(state['key'])
        seq_start = jrd.randint(subkey, (), 0, ep_len - sequence_length + 1)

        seq_start_index = lower + seq_start

        seq_slice = jax.ds(seq_start_index, sequence_length)

        # Return observations channels-last
        batch['observation'] = batch['observation'].at[i].set(
            jnp.moveaxis(
                # replay_buffer['observation'][seq_start: seq_end], 0, -1
                replay_buffer['observation'][seq_slice], 0, -1
            )
        )

        for k in ['action', 'reward', 'terminated']:
            batch[k] = batch[k].at[i].set(replay_buffer[k][seq_slice])

    state['batch'] = batch

    return state


def test_replay_buffer():
    from environments.replay_test import Env

    agent = Agent(Env, buffer_len=210, max_ep_len=256)

    timestep = {
        'observation': agent.env.reset(100)[0], 'action': 0, 'reward': 0,
        'terminated': False
    }

    agent.run_episode(timestep)

    agent.log()

    timestep['observation'] = agent.env.reset(60)[0]
    agent.run_episode(timestep)
    agent.log()

    num_ts = agent.state['replay_buffer']['timesteps_stored']

    print('act')
    print(agent.state['replay_buffer']['action'])

    assert num_ts == 160, num_ts


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


def get_action(obs, state):
    main_net = state['main_net']
    epsilon_interval = state['epsilon_interval']
    observations_collected = state['observations_collected']
    evaluating = state['evaluating']

    interval_progress = jnp.minimum(
        observations_collected / epsilon_interval, 1
    )

    epsilon = 1 - interval_progress * state['epsilon_range']

    epsilon = epsilon * jnp.float32(jnp.logical_not(evaluating))

    state['key'], subkey0, subkey1 = jrd.split(state['key'], num=3)

    take_random_action = jrd.uniform(subkey0) < epsilon

    action = jnp.where(
        take_random_action,
        jnp.argmax(main_net(obs[None, :])),
        jrd.randint(subkey1, (), 0, state['num_actions'])
    )

    state['observations_collected'] = observations_collected + 1

    return action, state


@jax.jit
def train(timestep, state):
    action, state = get_action(timestep['observation'], state)

    state['replay_buffer'] = add_timestep(timestep, state['replay_buffer'])

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
        'episode_probabilities': jnp.zeros(buffer_length, dtype=jnp.float32),
        'timesteps_stored': jnp.int32(0),
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
        self, env_constructor, buffer_len=1e6, max_ep_len=18e3, seq_len=2,
        min_epsilon=0.1
    ):
        self.log_items = {}
        self.max_ep_len = max_ep_len
        self.total_steps = 0

        env = env_constructor()
        obs_shape = env.observation_space.shape
        num_actions = env.action_space.n

        self.env = env

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
            'gamma': 0.99,
            'main_updates_since_target_update': 0,
            'main_updates_per_target_update': 100,
            'epsilon_range': 1 - min_epsilon,
            'epsilon_interval': 1_000,
            'observations_collected': 0,
            'evaluating': False,
        }

    def train(self, num_steps):
        initial_total_steps = self.total_steps
        while True:
            timestep = self.get_initial_timestep()
            self.run_episode(timestep)

            self.log()

            if self.total_steps >= initial_total_steps + num_steps:
                break

    def get_initial_timestep(self):
        obs, _ = self.env.reset()
        return {
            'observation': obs, 'action': 0, 'reward': 0,
            'terminated': False
        }

    def run_episode(self, timestep, evaluating=False):
        terminated = False
        episode_return = 0
        for i in range(self.max_ep_len):
            action, self.state = train(timestep, self.state)

            if terminated:
                break

            obs_next, reward, terminated, _, _ = self.env.step(action)

            timestep = {
                'observation': obs_next, 'action': action,
                'reward': reward, 'terminated': terminated
            }

            episode_return += reward

        self.total_steps += i + 1

        self.log_items['length'] = f'{i + 1:3}'
        self.log_items['return'] = f'{episode_return}'

        if evaluating:
            return episode_return

    def evaluate(self, num_episodes):
        self.state['evaluating'] = True
        total_return = 0

        for _ in range(num_episodes):
            timestep = self.get_initial_timestep()
            total_return += self.run_episode(timestep, evaluating=True)

        self.state['evaluating'] = False

        return total_return / num_episodes

    def log(self):
        log_item_strings = [
            key + ' ' + value for key, value in self.log_items.items()
        ]
        print(', '.join(log_item_strings))


if __name__ == '__main__':
    test_replay_buffer()
    exit()
    from environments.follow_the_dots import Env

    def get_env():
        return Env(3)

    agent = Agent(get_env, buffer_len=10_000, max_ep_len=100)

    num_eval_episodes = 40
    while True:
        agent.train(1_000)

        average_eval_return = agent.evaluate(num_eval_episodes)

        print(f'eval return {average_eval_return:.3}')

        if average_eval_return > (num_eval_episodes - 1) / num_eval_episodes:
            print('env solved')
            break
