import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.reward_shaping import reward_shaping  # Adjust import based on your filename

class TestRewardShaping(unittest.TestCase):
    def test_positive_reward(self):
        """Should return scaled positive reward when raw_reward > 0."""
        self.assertAlmostEqual(reward_shaping(50, old_lives=3, new_lives=3), 0.5)
        self.assertAlmostEqual(reward_shaping(10, old_lives=2, new_lives=2), 0.1)

    def test_life_lost_penalty(self):
        """Should return -0.5 when lives decrease."""
        self.assertEqual(reward_shaping(0, old_lives=3, new_lives=2), -0.5)
        self.assertEqual(reward_shaping(-5, old_lives=2, new_lives=1), -0.5)

    def test_small_default_reward(self):
        """Should return small positive reward when no points gained/lost and lives unchanged."""
        self.assertAlmostEqual(reward_shaping(0, old_lives=3, new_lives=3), 0.005)
        self.assertAlmostEqual(reward_shaping(-1, old_lives=2, new_lives=2), 0.005)

if __name__ == "__main__":
    unittest.main()
