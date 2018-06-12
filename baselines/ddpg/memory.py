import numpy as np
import pickle
import copy

from collections import deque


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations0 = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.terminals1 = RingBuffer(limit, shape=(1,))
        self.observations1 = RingBuffer(limit, shape=observation_shape)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs0_batch = self.observations0.get_batch(batch_idxs)
        obs1_batch = self.observations1.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        terminal1_batch = self.terminals1.get_batch(batch_idxs)

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        self.observations0.append(obs0)
        self.actions.append(action)
        self.rewards.append(reward)
        self.observations1.append(obs1)
        self.terminals1.append(terminal1)

    def save(self, pathname):
        pickle.dump([self.observations0, self.actions, self.rewards, self.terminals1, self.observations1],
                    open(pathname, 'wb'))

    def load(self, pathname):
        [self.observations0, self.actions, self.rewards, self.terminals1, self.observations1] = pickle.load(
            open(pathname, 'rb'))

    @property
    def nb_entries(self):
        return len(self.observations0)


class MemoryEpisodic(object):
    def __init__(self, limit, horizonlen, action_shape, observation_shape):
        self.limit = limit
        self.horizonlen = horizonlen
        self.currentMemEps = MemoryEps(horizonlen, action_shape, observation_shape)
        self.history = deque(maxlen=limit)

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 3, size=batch_size)

        obs0_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.observations0.shape[1]))
        obs1_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.observations1.shape[1]))
        action_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.actions.shape[1]))
        reward_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.rewards.shape[1]))
        terminal1_batch = np.zeros((batch_size, self.horizonlen, self.currentMemEps.terminals1.shape[1]))

        for k, i in zip(range(batch_size), batch_idxs):
            data = self.history[i].get_data()
            obs0_batch[k] = data['obs0']
            obs1_batch[k] = data['obs1']
            action_batch[k] = data['actions']
            reward_batch[k] = data['rewards']
            terminal1_batch[k] = data['terminals1']

        result = {
            'obs0': array_min2d(obs0_batch),
            'obs1': array_min2d(obs1_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'terminals1': array_min2d(terminal1_batch),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        self.currentMemEps.append(obs0, action, reward, obs1, terminal1, training)

    def append_episode(self):
        self.history.append(copy.copy(self.currentMemEps))
        self.currentMemEps.reset()

    def save(self, pathname):
        pickle.dump(self.history, open(pathname, 'wb'))

    def load(self, pathname):
        self.history = pickle.load(open(pathname, 'rb'))

    @property
    def nb_entries(self):
        return len(self.history)


class MemoryEps(object):
    def __init__(self, horizonlen, action_shape, observation_shape):
        self.horizonlen = horizonlen
        self.start = 0
        self.length = 0

        self.observations0 = np.zeros((horizonlen,) + observation_shape).astype('float32')
        self.actions = np.zeros((horizonlen,) + action_shape).astype('float32')
        self.rewards = np.zeros((horizonlen, 1,)).astype('float32')
        self.terminals1 = np.zeros((horizonlen, 1,)).astype('float32')
        self.observations1 = np.zeros((horizonlen,) + observation_shape).astype('float32')

    def reset(self):
        self.observations0.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.terminals1.fill(0)
        self.observations1.fill(0)

    def get_data(self):
        result = {
            'obs0': array_min2d(self.observations0),
            'obs1': array_min2d(self.observations1),
            'rewards': array_min2d(self.rewards),
            'actions': array_min2d(self.actions),
            'terminals1': array_min2d(self.terminals1),
        }
        return result

    def append(self, obs0, action, reward, obs1, terminal1, training=True):
        if not training:
            return

        if self.length < self.horizonlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.horizonlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.horizonlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.observations0[(self.start + self.length - 1) % self.horizonlen] = obs0
        self.actions[(self.start + self.length - 1) % self.horizonlen] = action
        self.rewards[(self.start + self.length - 1) % self.horizonlen] = reward
        self.observations1[(self.start + self.length - 1) % self.horizonlen] = obs1
        self.terminals1[(self.start + self.length - 1) % self.horizonlen] = terminal1

    @property
    def nb_entries(self):
        return len(self.observations0)
