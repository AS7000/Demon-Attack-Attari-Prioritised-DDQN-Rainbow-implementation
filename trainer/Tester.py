import sys
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.reward_shaping import reward_shaping


class Tester:
    def __init__(self, qnet, env, logger, episodes, action_size):
        self.qnet = qnet
        self.env = env
        self.logger = logger
        self.episodes = episodes
        self.action_size = action_size


    def test(self, weights_path):
        """Load weights and run test episodes."""
        self.qnet.model.load_weights(weights_path)

        for ep in range(self.episodes):
            self._test_episode(ep)

        # plot once at the end
        self.logger.plot_test()

    def _test_episode(self, episode_idx):
        state = self.env.reset()
        done = False
        raw_reward_total, scaled_reward_total = 0, 0
        while not done:
            action = self.select_action(state, 0.1)
            next_state, raw_reward, done = self.env.step(action)

            reward = reward_shaping(raw_reward)

            raw_reward_total += raw_reward
            scaled_reward_total += reward

            state = next_state

        stop = self.logger.log_metrics(
                episode=episode_idx,
                losses=[],
                q_vals=[],
                raw_reward=raw_reward_total,
                scaled_reward=scaled_reward_total,
                weights=0,
                model=self.qnet.model
            )

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        return int(np.argmax(self.qnet.predict(state)))