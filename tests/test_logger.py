import unittest
import numpy as np
import os
import sys
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import Logger

class DummyEnv:
    def __init__(self):
        self.reset_called = False
        self.step_count = 0
        self.env = self  
        self.action_space = self
        self.np_random = np.random.RandomState(42)

    def reset(self):
        self.reset_called = True
        self.step_count = 0
        return np.zeros((4,))  # dummy state

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 10
        return np.ones((4,)), 0, done, {}

    def sample(self):
        return 0  # dummy action

class DummyModel:
    def __init__(self):
        self.saved_paths = []

    def save_weights(self, path):
        open(path, 'a').close()  # touch file
        self.saved_paths.append(path)

    def predict(self, state_array, verbose=0):
        return np.array([[0.5, 1.0]])  # dummy Q-values

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.config = {
            "epsilon": 1.0,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.9,
            "project_name": "test_project"
        }
        self.env = DummyEnv()
        self.model = DummyModel()
        self.logger = Logger(
            config=self.config,
            env=self.env,
            model=self.model,
            save_path="temp_models",
            use_wandb=False
        )
        self.logger.collect_fixed_states(num_states=10)  # collect a few for diagnostics

    def tearDown(self):
        shutil.rmtree("temp_models", ignore_errors=True)

    def test_initial_epsilon_config(self):
        self.assertEqual(self.logger.epsilon, 1.0)
        self.assertEqual(self.logger.epsilon_min, 0.1)
        self.assertEqual(self.logger.epsilon_decay, 0.9)

    def test_collect_fixed_states(self):
        self.assertEqual(len(self.logger.fixed_states), 10)
        self.assertTrue(self.env.reset_called)

    def test_evaluate_q_on_fixed_states(self):
        avg_q = self.logger.evaluate_q_on_fixed_states()
        self.assertAlmostEqual(avg_q, 1.0)  # always returns max of [0.5, 1.0]

    def test_log_metrics_updates_lists(self):
        self.logger.log_metrics(
            episode=1,
            total_reward=100,
            losses=[1.0, 0.5],
            q_vals=[0.2, 0.8],
            raw_reward=120,
            scaled_reward=110,
            lives_lost=0,
            model=self.model
        )

        self.assertEqual(len(self.logger.episode_rewards), 1)
        self.assertEqual(self.logger.episode_rewards[0], 100)
        self.assertLessEqual(self.logger.epsilon, 1.0)

    def test_model_is_saved_on_best_q_values(self):
        episode = 1
        path = os.path.join("temp_models", f"best_model_ep{episode}.weights.h5")

        self.logger.log_metrics(
            episode=episode,
            total_reward=50,
            losses=[0.5],
            q_vals=[1.0],
            raw_reward=100,
            scaled_reward=80,
            lives_lost=0,
            model=self.model
        )

        self.assertTrue(os.path.exists(path))

    def test_multiple_logging_appends(self):
        for ep in range(1, 6):
            self.logger.log_metrics(
                episode=ep,
                total_reward=ep * 10,
                losses=[0.1 * ep, 0.2 * ep],
                q_vals=[0.5 * ep, 0.6 * ep],
                raw_reward=ep * 11,
                scaled_reward=ep * 9,
                lives_lost=ep % 2,
                model=self.model
            )

        self.assertEqual(len(self.logger.episode_rewards), 5)
        self.assertEqual(len(self.logger.mean_losses), 5)
        self.assertEqual(len(self.logger.mean_exp_return), 5)
        self.assertEqual(len(self.logger.q_eval_track), 5)
        self.assertEqual(len(self.logger.raw_rewards), 5)
        self.assertEqual(len(self.logger.scaled_rewards), 5)

        self.assertEqual(self.logger.episode_rewards[-1], 50)
        self.assertAlmostEqual(self.logger.mean_losses[-1], np.mean([0.5, 1.0]))
        self.assertAlmostEqual(self.logger.mean_exp_return[-1], np.mean([2.5, 3.0]))
        self.assertEqual(self.logger.q_eval_track[-1], 1.0)

if __name__ == "__main__":
    unittest.main()
