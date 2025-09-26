import unittest
import numpy as np
import tensorflow as tf
import random
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.Tester import Tester
from utils.logger import Logger
from envs.DemonAttackEnvironment import DemonAttackEnvironment
from models.Q_Network import Q_Network


class TestTester(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 10
        self.batch_size = 1
        self.episodes = 1
        self.input_shape = (84, 84, 4)
        self.action_size = 2
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.tau = 0.1
        self.optimiser = tf.keras.optimizers.Adam(self.learning_rate)
        self.update_frequency = 1
        self.mode = "test"
        self.weights_path = "saved_models/best_model_ep0.weights.h5"

        wandb_trigger = False

        self.env = DemonAttackEnvironment(None)
        self.qnet = Q_Network(
            self.input_shape,
            self.action_size,
            length_episodes=self.episodes,
            learning_rate=self.learning_rate,
            optimizer=self.optimiser
        )

        self.logger = Logger(
            config={
                "project_name": "DemonAttack_V2",
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "episodes": self.episodes,
                "input_shape": self.input_shape,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "learning_rate": self.learning_rate,
                "tau": self.tau,
                "optimizer": str(self.optimiser),
                "update_frequency": self.update_frequency,
                "mode": self.mode
            },
            env=self.env,
            model=self.qnet.model,
            use_wandb=wandb_trigger
        )
        self.logger.plot_test = lambda: None

        # Tester
        self.tester = Tester(
            self.qnet,
            self.env,
            self.logger,
            self.episodes,
            self.action_size
        )

    def test_loads_weights_and_runs_episode(self):
        if not os.path.exists(self.weights_path):
            self.skipTest(f"Weights file not found: {self.weights_path}")

        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

        self.tester.test(self.weights_path)

        self.assertEqual(len(self.tester.logger.q_eval_track), self.episodes)
        self.assertEqual(len(self.tester.logger.episode_rewards), self.episodes)

        for r in self.tester.logger.episode_rewards:
            self.assertIsInstance(r, (int, float, np.floating))

        for q in self.tester.logger.q_eval_track:
            self.assertIsInstance(q, (int, float, np.floating))


if __name__ == "__main__":
    unittest.main()
