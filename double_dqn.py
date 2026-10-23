from random import random, randint
from collections import defaultdict, namedtuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrd
from jax.tree_util import register_pytree_node_class
from flax import nnx
import optax


# Returns stale data if we try to get more than we have put.
def circular_get(buffer, get_index, put_index):
    # If incrementing get_index has set it equal to put_index then we have
    # emptied the buffer and put_index should be set to -1 to record this.
    new_get_index = (get_index + 1) % buffer.shape[0]
    new_put_index = jnp.where(
        new_get_index == put_index,
        jnp.int32(-1),
        put_index
    )
    return buffer[get_index], new_get_index, new_put_index


def circular_put(buffer, get_index, put_index, item):
    # Increment put_index, except for when it's equal to -1, which indicates
    # that the buffer is empty. In this case set it equal to the index after
    # get_index.
    new_put_index = jnp.where(
        put_index == -1,
        (get_index + 1) % buffer.shape[0],
        (put_index + 1) % buffer.shape[0]
    )

    # If put_index was -1 then leave get_index as it is. Otherwise if
    # new_put_index has reached get_index then we're now overwriting the
    # corresponding item, so increment get_index.
    new_get_index = jnp.where(
        jnp.logical_and(
            new_put_index == get_index, jnp.logical_not(put_index == -1)
        ),
        (get_index + 1) % buffer.shape[0],
        get_index
    )

    # If put_index was -1 write the item at get_index
    write_index = jnp.where(
        put_index == -1,
        get_index,
        put_index
    )

    return buffer.at[write_index].set(item), new_get_index, new_put_index


def write_timestep(i, replay_buffer):
    # replay_buffer, buffer_offset = data
    episode = replay_buffer['episode']
    timestep_buffer = replay_buffer['timestep_buffer']

    # for key in ['observation', 'action', 'reward', 'terminated']:
    #     replay_buffer[key] = replay_buffer[key].at[buffer_offset + i].set(
    #         replay_buffer['episode'][key][i]
    #     )

    timestep = Timestep(
        episode['observation'][i], episode['action'][i], episode['reward'][i],
        episode['terminal'][i]
    )

    timestep_buffer.put(timestep)

    # We have to return buffer_offset because a loop body has to return
    # something with the same pytree structure as what it receives.
    return replay_buffer


# TODO: rename replay_buffer['least_recent_bounds_index'] to something like
# boundaries_get_index
def update_episode_probabilities(replay_buffer):
    boundaries = replay_buffer['boundaries']
    add_bounds_index = replay_buffer['add_bounds_index']
    timesteps_stored = replay_buffer['timesteps_stored']
    episodes_stored = replay_buffer['episodes_stored']
    probs = replay_buffer['episode_probabilities']
    seq_len = replay_buffer['batch']['action'].shape[1]

    # TODO: BREAKS WHEN BOUNDARIES WRAPS
    least_recent_bounds_index = replay_buffer['least_recent_bounds_index'] - 1

    bounds_wrapped = add_bounds_index < least_recent_bounds_index

    bounds_wrapped = jnp.where(
        add_bounds_index == -1,
        False,
        bounds_wrapped
    )

    # jax.debug.print(
    #     'update_probs: least_recent_bounds_index: {a},
    #     ' add_bounds_index: {b}',
    #     a=least_recent_bounds_index, b=add_bounds_index
    # )

    for i, bounds in enumerate(boundaries):
        start = bounds[0]
        end = bounds[1]

        # Is the current index inside the slice containing the active episode
        # boundaries, assuming that the slice hasn't wrapped around to the
        # beginning of the buffer?
        inside = jnp.logical_and(
            i >= least_recent_bounds_index, i < add_bounds_index
        )

        # Or, if the slice has wrapped, is the index inside one of the two
        # slices that define the circular segment containing the active episode
        # boundaries?
        inside_wrapped = jnp.logical_or(
            i < add_bounds_index, i >= least_recent_bounds_index
        )

        set_prob = jnp.logical_or(
            jnp.logical_and(jnp.logical_not(bounds_wrapped), inside),
            jnp.logical_and(bounds_wrapped, inside_wrapped)
        )

        # Force this to be true after adding the very first episode
        first_ep = jnp.logical_and(
            episodes_stored == 1, add_bounds_index == -1
        )

        set_prob = jnp.where(
            jnp.logical_and(i == 0, first_ep),
            True,
            set_prob
        )

        # jax.lax.cond(
        #     set_prob,
        #     lambda s: jax.debug.print('set_prob true on step {a}', a=s),
        #     lambda s: None,
        #     i
        # )

        # jax.debug.print(
        #     'set_prob: {a}, episodes_stored: {b}, add_bounds_index: {c}',
        #     a=set_prob, b=episodes_stored, c=add_bounds_index
        # )

        # If seq_len is e.g. 2 then the last timestep can never occur as the
        # first timestep in a sequence. This means that the number of distinct
        # sequences that can be drawn from each episode is its length - 1.
        num_sequences = end - start - (seq_len - 1)
        total_sequences = timesteps_stored - episodes_stored * (seq_len - 1)

        prob = jnp.where(set_prob, num_sequences / total_sequences, 0)

        probs = probs.at[i].set(prob)

    replay_buffer['episode_probabilities'] = probs

    return replay_buffer


# def overlap_check(replay_buffer):
#     least_recent_ep_start = replay_buffer['least_recent_bounds'][0]

#     remove_ep = jnp.logical_and(
#         replay_buffer['wrapped'],
#         end_index > least_recent_ep_start
#     )

#     jax.debug.print(
#         'least_recent_bounds {a}, end_index {b}, iter_count {c},'
#         ' wrapped {d}, remove_ep {e}',
#         a=replay_buffer['least_recent_bounds'],
#         b=end_index,
#         c=replay_buffer['iter_count'],
#         d=replay_buffer['wrapped'],
#         e=remove_ep
#     )

#     return jnp.logical_and(
#         remove_ep,
#         replay_buffer['iter_count'] < 100
#     )


# def update_least_recent_bounds(replay_buffer):
#     # Use the current least_recent_bounds to decrement timesteps_stored
#     least_recent_bounds = replay_buffer['least_recent_bounds']
#     start = least_recent_bounds[0]
#     end = least_recent_bounds[1]

#     replay_buffer['timesteps_stored'] -= end - start
#     replay_buffer['episodes_stored'] -= 1

#     replay_buffer['iter_count'] += 1

#     (
#         least_recent_bounds, least_recent_bounds_index, add_bounds_index
#     ) = circular_get(
#         replay_buffer['boundaries'],
#         replay_buffer['least_recent_bounds_index'],
#         replay_buffer['add_bounds_index']
#     )

#     replay_buffer['least_recent_bounds'] = least_recent_bounds
#     replay_buffer['least_recent_bounds_index'] = least_recent_bounds_index

#     return replay_buffer


def add_episode(rb):
    # buffer_length = rb['observation'].shape[0]

    # length = rb['episode']['step_count']

    # boundaries = rb['boundaries']
    # add_bounds_index = rb['add_bounds_index']
    # least_recent_bounds_index = rb['least_recent_bounds_index']
    # start_index = rb['add_episode_index']

    # buffer_end_reached = start_index + length > buffer_length

    # # If there isn't enough space for the episode at the end of the buffer
    # # then put it at the beginning, otherwise put it directly after the
    # # previous episode.
    # start_index = jnp.where(buffer_end_reached, 0, start_index)

    # end_index = start_index + length

    # boundaries, least_recent_bounds_index, add_bounds_index = circular_put(
    #     boundaries, least_recent_bounds_index, add_bounds_index,
    #     [start_index, end_index]
    # )

    # # If we just added the first episode then get its bounds back out of the
    # # boundaries array and use them the initialise least_recent_bounds
    # (
    #     rb['least_recent_bounds'], rb['least_recent_bounds_index'],
    #     add_bounds_index,
    # ) = jax.lax.cond(
    #     rb['episodes_stored'] == 0,
    #     circular_get,
    #     lambda b, g, p: (rb['least_recent_bounds'], g, p),
    #     # Operands
    #     boundaries, least_recent_bounds_index, add_bounds_index
    # )

    # rb['boundaries'] = boundaries
    # rb['add_bounds_index'] = add_bounds_index
    # rb['add_episode_index'] = end_index

    start_index = rb['timestep_buffer'].put_index
    length = rb['episode']['step_count']

    # Write the timesteps to the main buffers
    rb = lax.fori_loop(0, length, write_timestep, rb)

    rb['slice_buffer'].put([start_index, length])

    # jax.debug.print(
    #     'just added ep: {a}, least_recent_bounds_index: {c}'
    #     ' add_bounds_index: {b}',
    #     a=replay_buffer['episodes_stored'],
    #     b=add_bounds_index,
    #     c=rb['least_recent_bounds_index']
    # )

    # rb['timesteps_stored'] += length
    # rb['episodes_stored'] += 1

    # # Record whether the buffer capacity has been reached so far
    # rb['wrapped'] = jnp.logical_or(
    #     rb['wrapped'], buffer_end_reached
    # )

    # rb['iter_count'] = 0

    # while_loop :: (a -> Bool) -> (a -> a) -> (a -> a)
    # rb = lax.while_loop(
    #     overlap_check,
    #     update_least_recent_bounds,
    #     rb
    # )

    # rb = update_episode_probabilities(rb)

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

    # for key in ['action', 'reward', 'terminated']:
    #     episode[key] = episode[key].at[prev_step_count].set(timestep[key])

    episode['step_count'] = step_count + 1

    return episode


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

    replay_buffer = lax.cond(
        should_add_episode,
        add_episode,
        lambda rb: rb,
        replay_buffer
    )

    return state


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


def inspect_metadata(rb):
    print(rb['action'])
    print(rb['boundaries'][:10])
    print(rb['add_episode_index'])
    print(rb['least_recent_bounds_index'])
    print(rb['least_recent_bounds'])


def inspect_batch(batch):
    for i in range(batch['action'].shape[0]):
        frame0 = batch['observation'][i, :, :, 0]
        frame1 = batch['observation'][i, :, :, 1]

        print(frame0)
        print(frame1)

        print(batch['action'][i], batch['reward'][i], batch['terminated'][i])
        print()


def test_batch_generation(agent, num_length_pairs):
    # We count the first timestep, which means that the last timestep of each
    # episode doesn't get included in the statistics.
    all_index_pairs = []
    for ep_num, length in num_length_pairs:
        all_index_pairs.extend([(ep_num, i) for i in range(length - 1)])

    pair_counts = defaultdict(int)

    batch_count = 0
    second_term = False
    conditions_satisfied = False
    iter_count = 0
    while not conditions_satisfied:
        agent.state = generate_batch(agent.state)
        batch = agent.state['replay_buffer']['batch']

        batch_count += 1
        for i, action_pair in enumerate(batch['action']):
            frame0 = batch['observation'][i, :, :, 0]
            frame1 = batch['observation'][i, :, :, 1]

            frame0_sum_gt_zero = frame0[0, 0] + frame0[0, 1] + frame0[1, 1] > 0

            # First frame data shouldn't all be zero, except for on the very
            # first frame, which we demark with a bottom-left value of 255.
            assert frame0_sum_gt_zero or frame0[1, 0] == 255

            # Second frame should never sum to zero
            assert jnp.sum(frame1) > 0

            # Both frames come from the same episode
            assert frame0[0, 0] == frame1[0, 0]

            # Second frame follows the first in its corresponding episode
            assert frame1[0, 1] == frame0[0, 1] + 1

            # Action we took in response to frame0 appears in frame1
            assert frame1[1, 0] == action_pair[0]

            term_pair = batch['terminated'][i]

            # Are either of the frames from the penultimate timestep of one of
            # the episodes? The -2 is because, for example, step 98 is the
            # penultimate step of an episode of length 100.
            penultimate_frame0 = any([
                list(frame0[0]) == [ep_num, length - 2]
                for ep_num, length in num_length_pairs
            ])

            penultimate_frame1 = any([
                list(frame1[0]) == [ep_num, length - 2]
                for ep_num, length in num_length_pairs
            ])

            # If frame0 is penultimate then its term value should be True,
            # otherwise its term value should be False.
            assert (
                (penultimate_frame0 and term_pair[0])
                or (not penultimate_frame0 and not term_pair[0])
            ), (frame0, term_pair)

            # Same for frame1
            assert (
                (penultimate_frame1 and term_pair[1])
                or (not penultimate_frame1 and not term_pair[1])
            ), (frame1[0, 1], term_pair[1])

            # Have we yet seen a second timestep with terminated == True?
            # (Just a minor sanity check)
            second_term = second_term or term_pair[1]

            pair_counts[(int(frame0[0, 0]), int(frame0[0, 1]))] += 1

        # Do all episode-timestep index pairs yet appear in our dict?
        all_pairs_found = sorted(pair_counts.keys()) == all_index_pairs

        total = 0
        ep_counts = defaultdict(int)

        for (episode_num, step_num), count in sorted(pair_counts.items()):
            total += count

            ep_counts[episode_num] += count

        ep_0_count, ep_1_count, ep_2_count = ep_counts.values()

        expected_prop_0 = (
            (num_length_pairs[0][1] - 1) / len(all_index_pairs)
        )

        expected_prop_1 = (
            (num_length_pairs[1][1] - 1) / len(all_index_pairs)
        )

        expected_prop_2 = (
            (num_length_pairs[2][1] - 1) / len(all_index_pairs)
        )

        ep_0_prop_correct = (ep_0_count / total) - expected_prop_0 < 0.03
        ep_1_prop_correct = (ep_1_count / total) - expected_prop_1 < 0.03
        ep_2_prop_correct = (ep_2_count / total) - expected_prop_2 < 0.03

        # Are timesteps from each episode appearing with a frequency
        # proportional to the length of the corresponding episode?
        props_correct = (
            ep_0_prop_correct and ep_1_prop_correct and ep_2_prop_correct
        )

        conditions_satisfied = (
            second_term and all_pairs_found and props_correct
        )

        iter_count += 1
        if iter_count >= 100:
            episode_probs = (
                agent.state['replay_buffer']['episode_probabilities']
            )

            last_prob = episode_probs[episode_probs.shape[0] - 1]

            assert second_term, second_term

            assert all_pairs_found, set(all_index_pairs) - set(pair_counts)

            assert props_correct, (
                f'episode_probs[:5]: {episode_probs[:10]}\n'
                f'episode_probs[-1]: {last_prob}\n'
                f'Expected probs: {expected_prop_0}, {expected_prop_1},'
                f' {expected_prop_2}\n'
                f'Actual probs: {ep_0_count / total}, {ep_1_count / total},'
                f' {ep_2_count / total}'
            )

            print(f'Too many iterations for {num_length_pairs}')

            assert iter_count < 100

    ep_0_prop = ep_0_count / total
    ep_1_prop = ep_1_count / total
    ep_2_prop = ep_2_count / total

    print(f'{batch_count} batches generated')
    print(f'Proportions: {ep_0_prop:.2} {ep_1_prop:.2} {ep_2_prop:.2}')


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

        # TODO: can we use lax.scan here?
        for i, (start_index, length) in enumerate(choices):
            seq_start = (
                start_index
                + jrd.randint(subkeys[i], (), 0, length - (self.seq_len - 1))
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
        self.buffer = jnp.zeros((length, 2), dtype=jnp.int32)
        self.get_index = jnp.int32(0)
        self.put_index = jnp.int32(0)
        self.empty = jnp.bool(True)
        self.probabilities = jnp.zeros(length, dtype=jnp.float32)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.update_probs = jnp.bool(False)

    def tree_flatten(self):
        children = (
            self.buffer, self.get_index, self.put_index, self.empty,
            self.probabilities, self.update_probs
        )
        aux_data = (self.batch_size, self.seq_len)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        instance = object.__new__(SliceBuffer)
        batch_size, seq_len = aux_data
        (
            buffer, get_index, put_index, empty, probabilities,
            update_probs
        ) = children

        instance.buffer = buffer
        instance.get_index = get_index
        instance.put_index = put_index
        instance.empty = empty
        instance.probabilities = probabilities
        instance.batch_size = batch_size
        instance.seq_len = seq_len
        instance.update_probs = update_probs

        return instance

    def overlap_check(self, item):
        most_recent_end = (item[0] + item[1]) % self.buffer.shape[0]

        least_recent_start = self.peek()[0]

        most_recent_end > least_recent_start

    def remove_overlapped_slices(self, item):
        pass

    def put(self, item):
        jax.debug.print('SliceBuffer: putting: {a}', a=item)
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

        lax.while_loop(
            self.overlap_check,
            self.remove_overlapped_slices,
            item
        )

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

    def peek_last(self):
        # Using mod creates the desired value even when put_index == 0
        return self.buffer[(self.put_index - 1) % self.buffer.shape[0]]

    def update_probabilities(self, probs):
        timesteps_stored = jnp.sum(self.buffer[:, 1])

        episodes_stored = jnp.where(
            self.put_index < self.get_index,
            self.put_index + self.buffer.shape[0] - self.get_index,
            self.put_index - self.get_index
        )

        # TODO: can we do this with simple array operations?
        for i in range(probs.shape[0]):
            probs = probs.at[i].set(
                jnp.maximum(0, (self.buffer[i, 1] - (self.seq_len - 1)))
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


def test_replay_buffer():
    from environments.replay_test import Env

    agent = Agent(Env, buffer_len=210, max_ep_len=200)

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

    num_ts = jnp.sum(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    print(agent.state['replay_buffer']['slice_buffer'].buffer[:, 1])
    assert num_ts == 200, num_ts

    test_batch_generation(agent, [(0, 100), (1, 60), (2, 40)])

    # TODO: test exactly full buffer
    # TODO: test wrapping boundaries array
    # TODO: test truncated episodes

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

    timestep = Timestep(
        agent.env.reset(length=80)[0], 0, 0, False
    )
    agent.run_episode(timestep)
    agent.log()

    test_batch_generation(agent, [(1, 60), (4, 70), (5, 80)])

    print('Assertions passed')


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


def get_regression_targets(batch, state):
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
        'observation': batch['observation'][:, :, :, :1],
        'action': batch['action'],
        'target': targets
    }

    loss, grads = loss_with_gradients(main_net, loss_batch)

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

    state['key'], subkey_0, subkey_1 = jrd.split(state['key'], num=3)

    action = lax.cond(
        jrd.uniform(subkey_0) < epsilon,
        lambda _: jrd.randint(subkey_1, (), 0, state['num_actions']),
        lambda observation: jnp.argmax(main_net(observation[None, :])),
        obs
    )

    state['observations_collected'] = observations_collected + 1

    return action, state


@jax.jit
def train(timestep, state):
    action, state = get_action(timestep.observation, state)

    state = add_timestep(timestep, state)

    state = update_nets(state)

    return action, state


def get_replay_buffer(
    length, max_episode_length, observation_shape, sequence_length
):
    # observation_buffer_shape = (length,) + observation_shape
    observation_episode_shape = (max_episode_length,) + observation_shape

    # The shortest an episode can be is two timesteps long
    max_episodes = jnp.int32(jnp.floor(length / 2))

    return {
        # 'observation': jnp.zeros(observation_buffer_shape, dtype=jnp.uint8),
        # 'action': jnp.zeros(buffer_length, dtype=jnp.int32),
        # 'reward': jnp.zeros(buffer_length, dtype=jnp.float32),
        # 'terminated': jnp.zeros(buffer_length, dtype=jnp.bool),
        'timestep_buffer': TimestepBuffer(
            length, observation_shape, sequence_length
        ),
        # 'boundaries': jnp.zeros((buffer_length, 2), dtype=jnp.int32),
        # 'episode_probabilities': jnp.zeros(buffer_length, dtype=jnp.float32),
        # 'timesteps_stored': jnp.int32(0),
        # 'episodes_stored': jnp.int32(0),
        # 'add_bounds_index': jnp.int32(-1),
        # 'least_recent_bounds_index': jnp.int32(0),
        # 'least_recent_bounds': jnp.zeros(2, dtype=jnp.int32),
        'slice_buffer': SliceBuffer(max_episodes, 32, sequence_length),
        # 'add_episode_index': jnp.int32(0),
        # 'wrapped': jnp.bool(0),
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
        # 'batch': {
        #     'observation': jnp.zeros(
        #         (32,) + observation_shape + (sequence_length,), dtype=jnp.uint8
        #     ),
        #     'action': jnp.zeros((32, 2), dtype=jnp.int32),
        #     'reward': jnp.zeros((32, 2), dtype=jnp.float32),
        #     'terminated': jnp.zeros((32, 2), dtype=jnp.bool)
        # },
        'iter_count': 0
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
        self, env_constructor, buffer_len=1_000_000, max_ep_len=18_000,
        seq_len=2, min_epsilon=0.1
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
        return Timestep(obs, 0, 0, False)

    def run_episode(self, timestep, evaluating=False):
        terminal = False
        episode_return = 0
        for i in range(self.max_ep_len):
            action, self.state = train(timestep, self.state)

            if terminal:
                break

            obs_next, reward, terminal, _, _ = self.env.step(action)

            timestep = Timestep(obs_next, action, reward, terminal)

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

    agent = Agent(get_env, buffer_len=1_000, max_ep_len=100)

    num_eval_episodes = 40
    while True:
        agent.train(1_000)

        average_eval_return = agent.evaluate(num_eval_episodes)

        print(f'eval return {average_eval_return:.3}')

        if average_eval_return > (num_eval_episodes - 1) / num_eval_episodes:
            print('env solved')
            break
