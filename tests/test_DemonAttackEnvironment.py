import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from envs.DemonAttackEnvironment import DemonAttackEnvironment  # Update with your actual filename
import gym


class TestDemonAttackEnvironment(unittest.TestCase):

    def setUp(self):
        """Set up environment for testing (no rendering)."""
        self.env = DemonAttackEnvironment(render_mode=None)

    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertIsInstance(self.env.env, gym.Env)
        self.assertEqual(self.env.state_dim, (4, 84, 84))
        self.assertTrue(isinstance(self.env.action_dim, int) and self.env.action_dim > 0)

    def test_reset_returns_correct_shape(self):
        """Test that reset returns a correctly shaped state."""
        state = self.env.reset()
        self.assertEqual(state.shape, (84, 84, 4))
        self.assertTrue(np.all(state >= 0.0) and np.all(state <= 1.0))

    def test_step_returns_correct_output(self):
        """Test that the step method returns expected values."""
        action = self.env.env.action_space.sample()
        state, reward, done, lives = self.env.step(action)

        self.assertEqual(state.shape, (84, 84, 4))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(lives, int)

    def test_preprocess_state_valid_input(self):
        """Test preprocessing on a valid RGB image."""
        raw_state, _ = self.env.env.reset()
        processed = self.env.preprocess_state(raw_state)
        self.assertEqual(processed.shape, (84, 84))
        self.assertTrue(np.issubdtype(processed.dtype, np.floating))

    def test_preprocess_state_invalid_input(self):
        """Test that invalid input to preprocess_state raises ValueError."""
        with self.assertRaises(ValueError):
            self.env.preprocess_state(np.zeros((84, 84)))  # Not a 3-channel image

    def test_get_state_returns_valid_raw_state(self):
        """Test that get_state returns a valid raw frame."""
        raw_state = self.env.get_state()
        self.assertIsInstance(raw_state, np.ndarray)
        self.assertEqual(raw_state.ndim, 3)

    def test_env_id_returns_string(self):
        """Test that env_id returns the correct environment ID string."""
        env_id = self.env.env_id()
        self.assertEqual(env_id, "DemonAttackNoFrameskip-v4")


if __name__ == "__main__":
    unittest.main()
