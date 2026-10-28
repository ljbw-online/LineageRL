from random import random, randint

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrd
from flax import nnx
import optax

from common import RunningStats
from replay_buffer import get_replay_buffer, Timestep


def write_timestep(i, replay_buffer):
    episode = replay_buffer['episode']
    timestep_buffer = replay_buffer['timestep_buffer']

    timestep = Timestep(
        episode['observation'][i], episode['action'][i], episode['reward'][i],
        episode['terminal'][i]
    )

    timestep_buffer.put(timestep)

    replay_buffer['timestep_buffer'] = timestep_buffer

    return replay_buffer


def add_episode(rb):
    start_index = rb['timestep_buffer'].put_index
    length = rb['episode']['step_count']

    # Write the timesteps to the main buffers
    rb = lax.fori_loop(0, length, write_timestep, rb)

    rb['slice_buffer'].put([start_index, length])

    rb['episode']['step_count'] = 0

    return rb


def add_timestep_to_episode(timestep, episode):
    step_count = episode['step_count']

    episode['observation'] = (
        episode['observation'].at[step_count].set(timestep.observation)
    )

    # Along with each observation we get the action, reward and terminated
    # value for the previous step. When step_count == 0 we just write
    # placeholders, and then write the actual values on the next step.
    prev_step_count = jnp.maximum(0, step_count - 1)

    episode['action'], episode['reward'], episode['terminal'] = (
        episode['action'].at[prev_step_count].set(timestep.action),
        episode['reward'].at[prev_step_count].set(timestep.reward),
        episode['terminal'].at[prev_step_count].set(timestep.terminal)
    )

    episode['step_count'] = step_count + 1

    return episode


@jax.jit  # Sometimes we call this directly during testing
def add_timestep(timestep, state):
    replay_buffer = state['replay_buffer']
    episode = replay_buffer['episode']

    # Jax clamps too-large indices but wraps negative indices. Calculate this
    # before step_count gets incremented.
    prev_term_index = jnp.maximum(0, episode['step_count'] - 1)

    # Increments step_count
    episode = add_timestep_to_episode(timestep, episode)

    # We reached the max episode length or we got terminated on the previous
    # step.
    should_add_episode = jnp.logical_or(
        episode['step_count'] == replay_buffer['max_episode_length'],
        episode['terminal'][prev_term_index]
    )

    state['replay_buffer'] = lax.cond(
        should_add_episode,
        add_episode,
        lambda rb: rb,
        replay_buffer
    )

    return state


def get_regression_targets(batch, state):
    main_net = state['main_net']
    target_net = state['target_net']

    next_step_obs = batch['observation'][:, 1]

    main_next_step_action_values = main_net(next_step_obs)
    target_next_step_action_values = target_net(next_step_obs)

    main_next_step_selected_actions = jnp.argmax(
        main_next_step_action_values, axis=1
    )

    target_values_of_main_actions = target_next_step_action_values[
        jnp.arange(next_step_obs.shape[0]), main_next_step_selected_actions
    ]

    rewards = batch['reward'][:, 0]
    terminal = batch['terminal'][:, 0]
    gamma = state['gamma']

    targets = (
        rewards
        + jnp.logical_not(terminal) * gamma * target_values_of_main_actions
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
    nnx.update(target_net, main_params)

    state['main_updates_since_target_update'] = 0

    return state


def update_nets(state):
    main_net = state['main_net']
    optimiser = state['optimiser']

    state['key'], subkey_0, subkey_1 = jrd.split(state['key'], num=3)

    choices = state['replay_buffer']['slice_buffer'].choice(subkey_0)
    batch = state['replay_buffer']['timestep_buffer'].get_batch(
        subkey_1, choices
    )
    targets = get_regression_targets(batch, state)

    loss_batch = {
        'observation': batch['observation'][:, 0],
        'action': batch['action'],
        'target': targets
    }

    loss, grads = loss_with_gradients(main_net, loss_batch)

    optimiser.update(main_net, grads)

    state['total_main_updates'] += 1
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
    total_main_updates = state['total_main_updates']
    initial_replay_steps = state['initial_replay_steps']
    # evaluating = state['evaluating']

    interval_progress = jnp.minimum(
        total_main_updates / epsilon_interval, 1
    )

    epsilon = jnp.where(
        state['evaluating'],
        state['eval_epsilon'],
        1 - interval_progress * state['epsilon_range']
    )

    # epsilon = epsilon * jnp.float32(jnp.logical_not(evaluating))

    state['key'], subkey_0, subkey_1 = jrd.split(state['key'], num=3)

    initial_replay = observations_collected < initial_replay_steps

    perform_random_action = jnp.logical_or(
        jrd.uniform(subkey_0) < epsilon,
        initial_replay
    )

    action = lax.cond(
        perform_random_action,
        lambda _: jrd.randint(subkey_1, (), 0, state['num_actions']),
        lambda observation: jnp.argmax(main_net(observation[None, :])),
        obs
    )

    state['observations_collected'] += 1

    return action, state


@jax.jit
def train(timestep, state):
    action, state = get_action(timestep.observation, state)

    state = add_timestep(timestep, state)

    initial_replay = (
        state['observations_collected'] < state['initial_replay_steps']
    )

    state = lax.cond(
        jnp.logical_or(state['evaluating'], initial_replay),
        lambda s: s,
        update_nets,
        state
    )

    return action, state


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
        self, env_runner, obs_q, buffer_len=250_000, max_ep_len=18_000,
        seq_len=2, main_updates_per_target_update=2_500, min_epsilon=0.1,
        epsilon_interval=250_000, initial_replay_steps=12_500, testing=False
    ):
        self.testing = testing
        self.log_items = {}
        self.max_ep_len = max_ep_len
        self.total_steps = 0
        self.initial_replay_steps = initial_replay_steps
        self.return_running_mean = RunningStats()

        self.obs_q = obs_q

        # self.env_runner = EnvRunner(self.obs_q)
        self.env_runner = env_runner

        obs_shape, num_actions = self.obs_q.get()

        # env = env_constructor()
        # obs_shape = env.observation_space.shape
        # num_actions = env.action_space.n

        self.num_actions = num_actions

        # del env

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
            'main_updates_per_target_update': main_updates_per_target_update,
            'epsilon_range': 1 - min_epsilon,
            'epsilon_interval': epsilon_interval,
            'observations_collected': 0,
            'loss': 0.0,
            'evaluating': False,
            'eval_epsilon': 0.05,
            'initial_replay_steps': initial_replay_steps,
            'total_main_updates': 0
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
        self.env_runner.reset()
        obs = self.obs_q.get()
        return Timestep(obs, 0, 0, False)

    def run_episode(self, timestep, evaluating=False):
        terminal = False
        episode_return = 0
        for i in range(self.max_ep_len):
            if self.testing:
                action = randint(0, self.num_actions)
                self.state = add_timestep(timestep, self.state)
            else:
                action, self.state = train(timestep, self.state)

            if terminal:
                break

            self.env_runner.put(int(action))
            obs_next, reward, terminal = self.obs_q.get()

            timestep = Timestep(obs_next, action, reward, terminal)

            episode_return += reward

        self.total_steps += i + 1

        self.log_items['timesteps collected'] = f'{self.total_steps:7}'
        self.log_items['length'] = f'{i + 1:5}'
        self.log_items['return'] = f'{episode_return}'
        self.log_items['average return'] = (
            f'{self.return_running_mean(episode_return):.2f}'
        )
        # self.log_items['observations_collected'] = (
        #     f'{self.state['observations_collected']:5}'
        # )
        # self.log_items['total_main_updates'] = (
        #     f'{self.state['total_main_updates']:5}'
        # )

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
