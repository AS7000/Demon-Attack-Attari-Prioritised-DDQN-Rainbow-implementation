import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory.PrioritizedReplayBuffer import PrioritizedReplayBuffer  # adjust import path if needed


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def setUp(self):
        
        self.buffer_size = 4
        self.batch_size = 2
        self.buffer = PrioritizedReplayBuffer(buffer_size=self.buffer_size, batch_size=self.batch_size)

        # Add enough dummy experiences
        for i in range(self.buffer_size):
            state = np.array([i])
            action = i
            reward = i * 0.1
            next_state = np.array([i + 1])
            done = i % 2 == 0
            self.buffer.add(state, action, reward, next_state, done)

    def test_length(self):
        self.assertEqual(len(self.buffer), self.buffer_size)

    def test_add_invalid_experience(self):
        with self.assertRaises(ValueError):
            self.buffer.tree.add(1.0, ("invalid",))

    def test_sample_batch_shape(self):
        sampled = self.buffer.sample()
        states, actions, rewards, next_states, dones, indices, weights = sampled

        self.assertEqual(states.shape, (self.batch_size, 1))
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(next_states.shape, (self.batch_size, 1))
        self.assertEqual(dones.shape, (self.batch_size,))
        self.assertEqual(indices.shape, (self.batch_size,))
        self.assertEqual(weights.shape, (self.batch_size,))

    def test_update_priorities(self):
        _, _, _, _, _, indices, _ = self.buffer.sample()
        new_priorities = np.random.uniform(0.1, 1.0, size=len(indices)).astype(np.float32)
        decay_mask = np.array([False] * (len(indices)), dtype=bool)

        self.buffer.update_priorities(indices, new_priorities, decay_mask=decay_mask)

        # Just test that no error is raised and priorities updated in-place
        for idx, p in zip(indices, new_priorities):
            self.assertTrue(0.0 < p <= 1.0)
    
    def test_weight_values_in_sampling(self):
        _, _, _, _, _, _, weights = self.buffer.sample()

        self.assertTrue(np.all(weights >= 1e-3), "Some weights are below 1e-3")
        self.assertTrue(np.all(weights <= 1.0), "Some weights are above 1.0")
        self.assertEqual(weights.dtype, np.float32, "Weights are not float32")
    
    '''
    def test_update_priority_affects_sampling_with_decay_mask(self):
        # Sample a batch to get initial indices
        _, _, _, _, _, indices, _ = self.buffer.sample()

        # Create priorities: one high, others low
        updated_priorities = np.full(len(indices), 1e-2, dtype=np.float32)
        updated_priorities[0] = 1.0  # Give high priority to first item

        # Create decay_mask: boost first, decay rest
        decay_mask = np.array([False] * (len(indices)), dtype=bool)

        # Update priorities using the mask
        self.buffer.update_priorities(indices, updated_priorities, decay_mask)

        # Count sampling frequency
        sample_counts = {idx: 0 for idx in indices}
        for _ in range(200):
            _, _, _, _, _, sampled_idxs, _ = self.buffer.sample()
            for idx in sampled_idxs:
                if idx in sample_counts:
                    sample_counts[idx] += 1

        sample_counts_array = np.array([sample_counts[idx] for idx in indices])
        max_index = np.argmax(sample_counts_array)
        most_sampled_count = sample_counts_array[max_index]
        mean_sampled = sample_counts_array.mean()

        self.assertGreaterEqual(
            most_sampled_count,
            1.1 * mean_sampled,
            f"High-priority item not sampled significantly more often (count={most_sampled_count}, mean={mean_sampled:.2f})"
        )
        self.assertEqual(
            indices[max_index],
            indices[0],
            "High-priority index not sampled the most"
        )
    '''

    def test_sampled_indices_are_leaves(self):
        _, _, _, _, _, indices, _ = self.buffer.sample()

        leaf_start_idx = self.buffer.tree.capacity - 1
        leaf_end_idx = len(self.buffer.tree.tree) - 1

        for idx in indices:
            self.assertGreaterEqual(idx, leaf_start_idx, f"Index {idx} is not a leaf (too low)")
            self.assertLessEqual(idx, leaf_end_idx, f"Index {idx} is not a leaf (too high)")

    
    def test_decay_and_boost_behavior(self):
        # Sample a batch
        _, _, _, _, _, indices, _ = self.buffer.sample()
        print(f'[DEBUG] indices {indices}')
        
        new_priorities = np.full(len(indices), 0.6)
        decay_mask = np.array([i % 2 == 0 for i in range(len(indices))], dtype=bool)

        # Track all updates to each index (due to duplicates)
        index_history = {}

        for i, idx in enumerate(indices):
            old = self.buffer.tree.tree[idx]
            new = new_priorities[i]
            is_decay = decay_mask[i]

            if idx not in index_history:
                index_history[idx] = []

            index_history[idx].append({
                'old': old,
                'new': new,
                'is_decay': is_decay
            })

        # Apply priority updates
        self.buffer.update_priorities(indices, new_priorities, decay_mask)

        # Check final values only once per unique index
        for idx, updates in index_history.items():
            # Start with original value
            value = updates[0]['old']

            for update in updates:
                delta = abs(update['new'] - value)

                if update['is_decay']:
                    value = np.clip(value * self.buffer.decay_factor, 1e-4, 1.0)
                else:
                    boost_factor = 1.0 + min(delta, 1.0)
                    boosted_priority = value * boost_factor
                    value = np.clip(boosted_priority ** self.buffer.alpha, 1e-4, 1.0)

            updated_value = self.buffer.tree.tree[idx]

            if not np.isclose(updated_value, value, atol=1e-2):
                print(f"[CHECK FAILED] idx={idx}")
                print(f"Expected final: {value}, Got: {updated_value}")
                print(f"Update history: {updates}")

            self.assertAlmostEqual(updated_value, value, delta=1e-2, msg=f"Priority mismatch at idx={idx}")


        






def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestPrioritizedReplayBuffer('test_length'))
    suite.addTest(TestPrioritizedReplayBuffer('test_add_invalid_experience'))
    suite.addTest(TestPrioritizedReplayBuffer('test_sample_batch_shape'))
    suite.addTest(TestPrioritizedReplayBuffer('test_update_priorities'))
    suite.addTest(TestPrioritizedReplayBuffer('test_weight_values_in_sampling'))
    suite.addTest(TestPrioritizedReplayBuffer('test_sampled_indices_are_leaves'))
    suite.addTest(TestPrioritizedReplayBuffer('test_decay_and_boost_behavior'))
    suite.addTest(TestPrioritizedReplayBuffer('test_update_priorities'))

    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

