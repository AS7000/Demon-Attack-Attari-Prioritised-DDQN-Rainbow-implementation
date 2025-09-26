import unittest
import numpy as np
import tensorflow as tf
import random
import sys,os
import numbers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trainer.Trainer import Trainer
from memory.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from utils.logger import Logger
from envs.DemonAttackEnvironment import DemonAttackEnvironment
from models.Q_Network import Q_Network

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.buffer_size = 10
        self.batch_size = 1
        self.episodes = 1
        self.input_shape = (84,84,4)  
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
        wandb_trigger = False

        self.env = DemonAttackEnvironment(None)
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size
        )
        self.qnet = Q_Network(self.input_shape, self.action_size, length_episodes=self.episodes, learning_rate=self.learning_rate ,optimizer=self.optimiser) 
        self.target_net = Q_Network(self.input_shape, self.action_size, length_episodes=self.episodes, learning_rate=self.learning_rate ,optimizer=self.optimiser)

        state = np.zeros(self.input_shape, dtype=np.float32)
        for _ in range(5):
            self.replay_buffer.add(state, 0, 1.0, state, False)

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

        # Trainer
        self.trainer = Trainer(
            self.env,
            self.episodes,
            self.replay_buffer,
            self.logger,
            self.epsilon,
            self.qnet,
            self.target_net,
            self.action_size,
            self.batch_size,
            self.gamma,
            self.tau,
            self.optimiser,
            self.update_frequency
        )

    def test_soft_update_target_network(self):
        q_weights = [np.ones_like(w) for w in self.qnet.model.get_weights()]
        t_weights = [np.zeros_like(w) for w in self.target_net.model.get_weights()]
        self.qnet.model.set_weights(q_weights)
        self.target_net.model.set_weights(t_weights)

        self.trainer.soft_update_target_network()
        updated_weights = self.target_net.model.get_weights()

        for upd_w, q_w, t_w in zip(updated_weights, q_weights, t_weights):
            expected = self.trainer.tau * q_w + (1 - self.trainer.tau) * t_w
            np.testing.assert_allclose(upd_w, expected, rtol=1e-6, err_msg="Soft update incorrect")


    def test_training_step_runs(self):
        loss, q_val = self.trainer.training_step()
        self.assertIsInstance(loss, numbers.Real)
        self.assertIsInstance(q_val, numbers.Real)

    def test_train_runs(self):
        random.seed(0)
        np.random.seed(0)
        tf.random.set_seed(0)

        self.trainer.train()
        # Just ensure logger has something recorded (your Logger implementation may differ)
        self.assertTrue(hasattr(self.logger, "mean_losses"))


if __name__ == "__main__":
    unittest.main()