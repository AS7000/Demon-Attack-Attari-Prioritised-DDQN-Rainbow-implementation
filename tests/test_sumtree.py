import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory.SumTree import SumTree


class TestSumTree(unittest.TestCase):

    def setUp(self):
        self.capacity = 4
        self.tree = SumTree(self.capacity)
        self.experiences = [
            (1, 0, 1.0, 2, False),
            (2, 1, 2.0, 3, False),
            (3, 0, 0.5, 1, True),
            (4, 1, 0.1, 0, False)
        ]
        self.priorities = [1.0, 2.0, 0.5, 0.1]

    def test_initialization(self):
        self.assertEqual(len(self.tree), 0)
        self.assertEqual(self.tree.tree.shape[0], 2 * self.capacity - 1)
        self.assertEqual(self.tree.data.shape[0], self.capacity)

    def test_add_and_len(self):
        for i in range(len(self.experiences)):
            self.tree.add(self.priorities[i], self.experiences[i])
            self.assertEqual(len(self.tree), i + 1)

    def test_add_invalid_data(self):
        with self.assertRaises(ValueError):
            self.tree.add(1.0, "invalid")

        with self.assertRaises(ValueError):
            self.tree.add(1.0, (1, 2))  # too short

    def test_total_priority(self):
        for p, d in zip(self.priorities, self.experiences):
            self.tree.add(p, d)
        self.assertAlmostEqual(self.tree.total_priority(), sum(self.priorities), places=5)

    def test_update_priority(self):
        for p, d in zip(self.priorities, self.experiences):
            self.tree.add(p, d)

        tree_idx = self.capacity - 1  # first leaf
        new_priority = 3.0
        old_total = self.tree.total_priority()
        change = new_priority - self.tree.tree[tree_idx]

        self.tree.update(tree_idx, new_priority)
        self.assertAlmostEqual(self.tree.tree[0], old_total + change, places=5)

    def test_sample_leaf(self):
        for p, d in zip(self.priorities, self.experiences):
            self.tree.add(p, d)

        total = self.tree.total_priority()
        for _ in range(100):
            v = np.random.uniform(0, total)
            idx, p, d = self.tree.sample_leaf(v)
            self.assertTrue(0 <= idx < len(self.tree.tree))
            self.assertIn(d, self.experiences)
            self.assertGreater(p, 0)

    def test_data_overwrite(self):
        inserted_data = []
        for i in range(self.capacity * 2):  # force overwrite
            d = (i, i, i, i, False)
            self.tree.add(1.0, d)
            inserted_data.append(d)

        expected_data = inserted_data[-self.capacity:]
        
        self.assertEqual(len(self.tree), self.capacity)

        data_list = self.tree.data.tolist()

        for d in expected_data:
            self.assertIn(d, data_list)


    def test_minimum_total_priority(self):
        empty_tree = SumTree(4)
        self.assertGreater(empty_tree.total_priority(), 0)

        empty_tree.add(0.0, self.experiences[0])
        self.assertGreater(empty_tree.total_priority(), 0)


if __name__ == '__main__':
    unittest.main()
