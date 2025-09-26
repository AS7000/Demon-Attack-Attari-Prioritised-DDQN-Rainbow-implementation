import numpy as np
import random
from memory.SumTree import SumTree
import tensorflow as tf

class PrioritizedReplayBuffer:
    def __init__(self, state_size, buffer_size, batch_size,
                 alpha=0.6, beta=0.4, beta_increment=1e-4, epsilon=1e-6):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # PER params
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 2.0

        # 👉 Use NumPy arrays (much faster than tf.Variable for storage)
        self.state      = np.zeros((buffer_size, *state_size), dtype=np.float32)
        self.action     = np.zeros((buffer_size,), dtype=np.int32)
        self.reward     = np.zeros((buffer_size,), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, *state_size), dtype=np.float32)
        self.done       = np.zeros((buffer_size,), dtype=np.float32)

        self.count = 0
        self.real_size = 0

    def add(self, state, action, reward, next_state, done):
        idx = self.count
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, idx)

        # 👉 direct NumPy assignment (no tf ops)
        self.state[idx]      = state
        self.action[idx]     = action
        self.reward[idx]     = reward
        self.next_state[idx] = next_state
        self.done[idx]       = done

        self.count = (self.count + 1) % self.buffer_size
        self.real_size = min(self.real_size + 1, self.buffer_size)

    def sample(self):
        assert self.real_size >= self.batch_size

        indices = np.empty((self.batch_size,), dtype=np.int32)
        priorities = np.empty((self.batch_size,), dtype=np.float32)
        tree_indices = np.empty((self.batch_size,), dtype=np.int32)

        segment = self.tree.total_priority() / self.batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(self.batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data_idx = self.tree.sample_leaf(s)
            indices[i] = data_idx
            priorities[i] = priority
            tree_indices[i] = idx

        probs = priorities / (self.tree.total_priority() + self.epsilon)
        weights = (self.real_size * probs) ** (-self.beta)
        weights /= (weights.max() + 1e-8)

        # 👉 Convert only the sampled batch to tensors (GPU ready)
        states      = tf.convert_to_tensor(self.state[indices], dtype=tf.float32)
        actions     = tf.convert_to_tensor(self.action[indices], dtype=tf.int32)
        rewards     = tf.convert_to_tensor(self.reward[indices], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_state[indices], dtype=tf.float32)
        dones       = tf.convert_to_tensor(self.done[indices], dtype=tf.float32)
        weights     = tf.convert_to_tensor(weights, dtype=tf.float32)

        batch = (states, actions, rewards, next_states, dones)
        return batch, weights, tree_indices

    def update_priorities(self, indices, priorities):
        # 👉 Vectorized update
        adjusted = (np.abs(priorities) + self.epsilon) ** self.alpha
        for idx, p in zip(indices, adjusted):
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def can_learn(self):
        return self.real_size >= self.batch_size and self.real_size >= (0.1 * self.buffer_size)
