from collections import defaultdict, namedtuple

import jax
import jax.numpy as jnp
import jax.random as jrd
from jax.tree_util import register_pytree_node_class
import jax.lax as lax


@jax.jit
def test_slice_buffer():
    pairs = SliceBuffer(5, 10, 2)

    pairs.put([1, 2])
    jax.debug.print('After first put: empty {a}', a=pairs.empty)

    item = pairs.get()
    jax.debug.print(
        'After first get:\n{a}, empty {b}',
        a=pairs.buffer, b=pairs.empty
    )

    for item in [[3, 4], [5, 6], [7, 8], [9, 10]]:
        pairs.put(item)

    jax.debug.print(
        'Buffer almost full:\n{a}, empty {b}, get_index {c}, put_index {d}',
        a=pairs.buffer, b=pairs.empty, c=pairs.get_index, d=pairs.put_index
    )

    pairs.put([11, 12])
    jax.debug.print(
        'Full:\n{a}, empty {b}, get_index {c}, put_index {d}',
        a=pairs.buffer, b=pairs.empty, c=pairs.get_index, d=pairs.put_index
    )

    pairs.put([13, 14])
    jax.debug.print(
        '3, 4 overwritten:\n{a}, empty {b}',
        a=pairs.buffer, b=pairs.empty
    )

    item = pairs.get()
    jax.debug.print(
        '5, 6 erased:\n{a}, empty {b}, item {c}',
        a=pairs.buffer, b=pairs.empty, c=item
    )

    key = jrd.key(57)
    key, subkey = jrd.split(key)

    choices = pairs.choice(subkey)
    jax.debug.print('choices:\n{a}', a=choices)

    key, subkey = jrd.split(key)
    choices = pairs.choice(subkey)

    jax.debug.print('choices:\n{a}', a=choices)

    jax.debug.print('probs: {a}', a=pairs.probabilities)


@jax.jit
def get_batch_for_test(state):
    state['key'], subkey_0, subkey_1 = jrd.split(
        state['key'], num=3
    )

    choices = state['replay_buffer']['slice_buffer'].choice(subkey_0)
    batch = state['replay_buffer']['timestep_buffer'].get_batch(
        subkey_1, choices
    )

    return batch, state


def test_batch_generation(agent, num_length_pairs):
    timestep_buffer = agent.state['replay_buffer']['timestep_buffer']

    ep_nums = [i for i, _ in num_length_pairs]

    max_ep_len = agent.state['replay_buffer']['max_episode_length']

    truncated_eps = dict()
    for num, length in num_length_pairs:
        truncated_eps[num] = length > max_ep_len

    # Clip to max episode length
    num_length_pairs = [(i, min(j, max_ep_len)) for i, j in num_length_pairs]

    # We count the first timestep of each sequence, which means that the last
    # timestep of each episode doesn't get included in the statistics.
    all_index_pairs = []
    for ep_num, length in num_length_pairs:
        all_index_pairs.extend([(ep_num, i) for i in range(length - 1)])

    pair_counts = defaultdict(int)

    batch_count = 0
    conditions_satisfied = False
    iter_count = 0
    while not conditions_satisfied:
        batch, agent.state = get_batch_for_test(agent.state)

        batch_count += 1
        for i, action_array in enumerate(batch['action']):
            frame0 = batch['observation'][i, 0]
            frame1 = batch['observation'][i, 1]

            frame0_sum_gt_zero = frame0[0, 0] + frame0[0, 1] + frame0[1, 1] > 0

            # First frame data shouldn't all be zero, except for on the very
            # first frame, which we demark with a bottom-left value of 255.
            assert frame0_sum_gt_zero or frame0[1, 0] == 255

            # Frame is from an episode which is still in the replay buffer
            assert frame0[0, 0] in ep_nums, (
                frame0,
                ep_nums,
                timestep_buffer.observation[:, 0, 0]
            )

            # Second frame should never sum to zero
            assert jnp.sum(frame1) > 0

            # Both frames come from the same episode
            assert frame0[0, 0] == frame1[0, 0], (frame0, frame1)

            # Second frame follows the first in its corresponding episode
            assert frame1[0, 1] == frame0[0, 1] + 1, (frame0, frame1)

            # Action we took in response to frame0 appears in frame1
            assert frame1[1, 0] == action_array[0], (
                frame1,
                action_array[0],
                timestep_buffer.observation[:, 0],
                timestep_buffer.action
            )

            terminal = batch['terminal'][i]

            # Is the first frame from the penultimate timestep of one
            # of the episodes? The -2 is because, for example, step 98 is the
            # penultimate step of an episode of length 100.
            penultimate_frame0 = any([
                list(frame0[0]) == [ep_num, length - 2]
                for ep_num, length in num_length_pairs
            ])

            truncated = truncated_eps[int(frame0[0, 0])]

            if penultimate_frame0 and not truncated:
                assert terminal[0], (frame0, truncated, terminal)
            else:
                assert not terminal[0], (frame0, truncated, terminal)

            pair_counts[(int(frame0[0, 0]), int(frame0[0, 1]))] += 1

        # Do all episode-timestep index pairs yet appear in our dict?
        all_pairs_found = sorted(pair_counts.keys()) == all_index_pairs

        total = 0
        ep_counts = defaultdict(int)

        for (episode_num, step_num), count in sorted(pair_counts.items()):
            total += count
            ep_counts[episode_num] += count

        expected_props = [
            (length - 1) / len(all_index_pairs)
            for _, length in num_length_pairs
        ]

        actual_props = [
            count / total for count in ep_counts.values()
        ]

        # Are timesteps appearing with a frequency proportional to the length
        # of their corresponding episode?
        props_correct = all([
            abs(actual - expected) < 0.03
            for actual, expected in zip(actual_props, expected_props)
        ])

        conditions_satisfied = all_pairs_found and props_correct

        iter_count += 1
        if iter_count >= 100:
            print(f'Too many iterations for {num_length_pairs}')

            assert all_pairs_found, (
                sorted(set(all_index_pairs) - set(pair_counts))
            )

            assert props_correct, (
                f'Expected probs: {expected_props}\n'
                f'Actual probs: {actual_props}'
            )

            assert iter_count < 100

    print(f'{batch_count} batches generated')
    print(
        'Proportions:',
        ', '.join([f'{p:.2}' for p in actual_props])
    )


def test_replay_buffer():
    from environments.replay_test import Env
    from train import Agent

    agent = Agent(Env, buffer_len=210, max_ep_len=104, testing=True)

    timestep = Timestep(
        agent.env.reset(length=100)[0], 0, 0, False
    )

    agent.run_episode(timestep)
    agent.log()

    timestep = Timestep(
        agent.env.reset(length=60)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    timestep = Timestep(
        agent.env.reset(length=40)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    print(agent.state['replay_buffer']['slice_buffer'].buffer[:10])

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 200, num_ts

    test_batch_generation(agent, [(0, 100), (1, 60), (2, 40)])

    timestep = Timestep(
        agent.env.reset(length=40)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 140, num_ts

    test_batch_generation(agent, [(1, 60), (2, 40), (3, 40)])

    timestep = Timestep(
        agent.env.reset(length=70)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 210, num_ts

    test_batch_generation(agent, [(1, 60), (2, 40), (3, 40), (4, 70)])

    timestep = Timestep(
        agent.env.reset(length=80)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 190, num_ts

    test_batch_generation(agent, [(3, 40), (4, 70), (5, 80)])

    timestep = Timestep(
        agent.env.reset(length=20)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    # Fill the replay buffer up exactly. This makes it easier to calculate what
    # the episode numbers should be in the following code.
    test_batch_generation(agent, [(3, 40), (4, 70), (5, 80), (6, 20)])
    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 210, num_ts

    print('slices', agent.state['replay_buffer']['slice_buffer'].buffer[:10])

    num_slice_wrap_eps = 100

    # Write lots of episodes in order to wrap the slice buffer and check that
    # batches are still correct after this.
    print(f'running {num_slice_wrap_eps} eps')
    for _ in range(num_slice_wrap_eps):
        timestep = Timestep(
            agent.env.reset(length=21)[0], 0, 0, False
        )
        agent.run_episode(timestep)
        # agent.log()

    print('done')

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 210, num_ts

    num_length_pairs = [
        (i, 21)
        for i in range(7 + num_slice_wrap_eps - 10, 7 + num_slice_wrap_eps)
    ]

    test_batch_generation(agent, num_length_pairs)

    timestep = Timestep(
        agent.env.reset(length=120)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    print('num_length_pairs', num_length_pairs)

    sb = agent.state['replay_buffer']['slice_buffer'].buffer
    print('slices:')
    for start, length in sb:
        if start != 0:
            print(start, length)

    # We tried to write an ep of len 120 but max_ep_len is 110 so we only write
    # one of 110 steps.
    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 209, num_ts

    # Again the actual length written to the buffer is 110 even though we tried
    # 120
    test_batch_generation(
        agent, [
            (102, 21), (103, 21), (104, 21), (105, 21), (106, 21), (107, 120)
        ]
    )

    timestep = Timestep(
        agent.env.reset(length=120)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    timestep = Timestep(
        agent.env.reset(length=120)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    test_batch_generation(
        agent, [(108, 120), (109, 120)]
    )

    print('Assertions passed')


def get_replay_buffer(
    length, max_episode_length, observation_shape, sequence_length
):
    observation_episode_shape = (max_episode_length,) + observation_shape

    return {
        'timestep_buffer': TimestepBuffer(
            length, observation_shape, sequence_length
        ),
        'slice_buffer': SliceBuffer(length, 32, sequence_length),
        'max_episode_length': jnp.int32(max_episode_length),
        'episode': {
            'observation': jnp.zeros(
                observation_episode_shape, dtype=jnp.uint8
            ),
            'action': jnp.zeros(max_episode_length, dtype=jnp.int32),
            'reward': jnp.zeros(max_episode_length, dtype=jnp.float32),
            'terminal': jnp.zeros(max_episode_length, dtype=jnp.bool),
            'step_count': jnp.int32(0),
        },
        'iter_count': 0
    }


Timestep = namedtuple(
    'Timestep', ['observation', 'action', 'reward', 'terminal']
)


@register_pytree_node_class
class TimestepBuffer:
    def __init__(self, length, obs_shape, seq_len):
        self.observation = jnp.zeros((length,) + obs_shape, dtype=jnp.uint8)
        self.action = jnp.zeros(length, dtype=jnp.int32)
        self.reward = jnp.zeros(length, dtype=jnp.float32)
        self.terminal = jnp.zeros(length, dtype=jnp.bool)

        self.get_index = jnp.int32(0)
        self.put_index = jnp.int32(0)
        self.empty = jnp.bool(True)
        self.seq_len = seq_len

    def tree_flatten(self):
        children = (
            self.observation, self.action, self.reward, self.terminal,
            self.get_index, self.put_index, self.empty
        )
        aux_data = (self.seq_len,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = object.__new__(TimestepBuffer)
        (seq_len,) = aux_data
        (
            observation, action, reward, terminal, get_index, put_index, empty
        ) = children

        instance.observation = observation
        instance.action = action
        instance.reward = reward
        instance.terminal = terminal
        instance.get_index = get_index
        instance.put_index = put_index
        instance.empty = empty
        instance.seq_len = seq_len

        return instance

    def put(self, timestep):
        self.observation = self.observation.at[self.put_index].set(
            timestep.observation
        )
        self.action = self.action.at[self.put_index].set(timestep.action)
        self.reward = self.reward.at[self.put_index].set(timestep.reward)
        self.terminal = self.terminal.at[self.put_index].set(timestep.terminal)

        increment_get = jnp.logical_and(
            self.put_index == self.get_index, jnp.logical_not(self.empty)
        )

        # If put_index is equal to get_index and the buffer isn't empty
        # then we're now overwriting the least recent item, so increment
        # get_index. Alternately if the buffer is empty then leave get_index as
        # it is.
        self.get_index = jnp.where(
            increment_get,
            (self.get_index + 1) % self.action.shape[0],
            self.get_index
        )

        self.put_index = (self.put_index + 1) % self.action.shape[0]
        self.empty = False

    def get_batch(self, key, choices):
        batch_obs_shape = (
            (choices.shape[0], self.seq_len,) + self.observation.shape[1:]
        )

        batch_scalars_shape = (choices.shape[0], self.seq_len - 1)

        batch = {
            'observation': jnp.zeros(batch_obs_shape, dtype=jnp.uint8),
            'action': jnp.zeros(batch_scalars_shape, dtype=jnp.int32),
            'reward': jnp.zeros(batch_scalars_shape, dtype=jnp.float32),
            'terminal': jnp.zeros(batch_scalars_shape, dtype=jnp.bool)
        }

        subkeys = jrd.split(key, num=choices.shape[0])

        # TODO: reimplement with vmap
        for i, (ep_start_index, length) in enumerate(choices):
            random_step_count = jrd.randint(
                subkeys[i], (), 0, length - (self.seq_len - 1)
            )

            seq_start = (
                (ep_start_index + random_step_count)
                % self.observation.shape[0]
            )

            for j in range(self.seq_len):
                obs_index = (seq_start + j) % self.observation.shape[0]
                batch['observation'] = batch['observation'].at[i, j].set(
                    self.observation[obs_index]
                )

            batch['action'] = batch['action'].at[i].set(self.action[seq_start])
            batch['reward'] = batch['reward'].at[i].set(self.reward[seq_start])
            batch['terminal'] = batch['terminal'].at[i].set(
                self.terminal[seq_start]
            )

        return batch


@register_pytree_node_class
class SliceBuffer:
    def __init__(self, length, batch_size, seq_len):
        # The shortest an episode can be is two timesteps long
        max_episodes = jnp.int32(jnp.floor(length / 2))

        self.buffer = jnp.zeros((max_episodes, 2), dtype=jnp.int32)
        self.replay_buffer_length = length
        self.get_index = jnp.int32(0)
        self.put_index = jnp.int32(0)
        self.empty = jnp.bool(True)
        self.probabilities = jnp.zeros(max_episodes, dtype=jnp.float32)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.update_probs = jnp.bool(False)

    def tree_flatten(self):
        children = (
            self.buffer, self.get_index, self.put_index, self.empty,
            self.probabilities, self.update_probs
        )
        aux_data = (self.replay_buffer_length, self.batch_size, self.seq_len)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = object.__new__(SliceBuffer)
        replay_buffer_length, batch_size, seq_len = aux_data
        (
            buffer, get_index, put_index, empty, probabilities,
            update_probs
        ) = children

        instance.buffer = buffer
        instance.replay_buffer_length = replay_buffer_length
        instance.get_index = get_index
        instance.put_index = put_index
        instance.empty = empty
        instance.probabilities = probabilities
        instance.batch_size = batch_size
        instance.seq_len = seq_len
        instance.update_probs = update_probs

        return instance

    @staticmethod
    def overlap_check(pair):
        item, slice_buffer = pair
        most_recent_start = item[0]
        most_recent_end = (
            (most_recent_start + item[1]) % slice_buffer.replay_buffer_length
        )

        least_recent_start = slice_buffer.peek()[0]

        # The episode has been written across the array boundary, i.e. partly
        # at the end of the array and partly at the beginning.
        episode_wrapped = most_recent_end < most_recent_start

        # If the episode hasn't wrapped then an overlap is indicated by the
        # start of the least recently added episode being between the start and
        # the end of the episode currently being added.
        non_wrapped_overlap = jnp.logical_and(
            jnp.logical_not(episode_wrapped),
            jnp.logical_and(
                least_recent_start >= most_recent_start,
                least_recent_start < most_recent_end
            )
        )

        # If the episode has wrapped then an overlap is indicated by the start
        # of the least recently added episode either being less than the end of
        # the episode being added or greater than the start of the episode
        # being added.
        wrapped_overlap = jnp.logical_and(
            episode_wrapped,
            jnp.logical_or(
                least_recent_start < most_recent_end,
                least_recent_start >= most_recent_start
            )
        )

        least_recent_overlapped = jnp.logical_or(
            non_wrapped_overlap, wrapped_overlap
        )

        return jnp.logical_and(
            jnp.logical_not(slice_buffer.empty),
            least_recent_overlapped
        )

    @staticmethod
    def remove_overlapped_slice(pair):
        item, self_copy = pair
        self_copy.get()
        return item, self_copy

    def put(self, item):
        item = jnp.array(item, dtype=jnp.int32)

        self.buffer = self.buffer.at[self.put_index].set(item)

        increment_get = jnp.logical_and(
            self.put_index == self.get_index, jnp.logical_not(self.empty)
        )

        # If put_index is equal to get_index and the buffer isn't empty
        # then we're now overwriting the least recent item, so increment
        # get_index. Alternately if the buffer is empty then leave get_index as
        # it is.
        self.get_index = jnp.where(
            increment_get,
            (self.get_index + 1) % self.buffer.shape[0],
            self.get_index
        )

        # Remove slices corresponding to episodes that are being wholly or
        # partly overwritten by the current episode.
        (removed_item, modified_self) = lax.while_loop(
            self.overlap_check,
            self.remove_overlapped_slice,
            (item, self)
        )

        # Inside a JITed function, if we modify an object by calling a method
        # on it then we have to return the object from the function, otherwise
        # the modification was a side effect. So to use while_loop we have to
        # pass self to it and then return the instance at the end. Calling the
        # the thing returned by while_loop "self" doesn't seem to work; we have
        # to manually transfer the modified attributes.
        self.get_index = modified_self.get_index
        self.buffer = modified_self.buffer
        self.empty = modified_self.empty
        self.update_probs = modified_self.update_probs

        self.put_index = (self.put_index + 1) % self.buffer.shape[0]
        self.empty = False
        self.update_probs = True

    def get(self):
        item = self.buffer[self.get_index]

        # Erase the removed item to make sample probabilities easier to
        # calculate later.
        self.buffer = self.buffer.at[self.get_index].set(0)

        self.get_index = (self.get_index + 1) % self.buffer.shape[0]

        self.empty = self.get_index == self.put_index
        self.update_probs = True

        return item

    def peek(self):
        return self.buffer[self.get_index]

    def update_probabilities(self, probs):
        timesteps_stored = jnp.sum(self.buffer[:, 1])

        episodes_stored = jnp.where(
            self.put_index < self.get_index,
            self.put_index + self.buffer.shape[0] - self.get_index,
            self.put_index - self.get_index
        )

        probs = (
            jnp.maximum(0, self.buffer[:, 1] - (self.seq_len - 1))
            / (timesteps_stored - episodes_stored * (self.seq_len - 1))
        )

        self.update_probs = False

        return probs

    def choice(self, subkey):
        # If items have been added or removed since this method was last called
        # then update the sampling probabilities.
        self.probabilities = jax.lax.cond(
            self.update_probs,
            self.update_probabilities,
            lambda p: p,
            self.probabilities
        )

        return jrd.choice(
            subkey, self.buffer, shape=(self.batch_size,), p=self.probabilities
        )
