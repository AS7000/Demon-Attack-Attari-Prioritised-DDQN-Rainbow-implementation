import random
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.reward_shaping import reward_shaping
from utils.rewardnormaliser import RewardNormalizer


class Trainer:
    def __init__(self, env, episodes, replay_buffer, logger, epsilon,epsilon_min,epsilon_decay, qnet,
                 target_net, action_size, batch_size, gamma, tau, optimiser, update_frequency ):
        self.episodes = episodes
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.env = env
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.qnet = qnet
        self.target_net = target_net
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.optimiser = optimiser
        self.update_frequency = update_frequency
        self.rewardnormaliser = RewardNormalizer()
        self.steps = 0


    
    def train(self):
        for episode in range(self.episodes):
            stop = self._train_episode(episode)
            if stop:
                break
        if self.logger.use_wandb == True:
            self.logger.close()
        
    def _train_episode(self, episode):
        state = self.env.reset()
        done = False
        losses, q_vals = [], []
        raw_reward_total, scaled_reward_total = 0, 0
        stop = False
        weights = 0

        while not done:
            action = self.select_action(state, self.epsilon)
            next_state, raw_reward, done = self.env.step(action)

            reward = reward_shaping(raw_reward)
            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            scaled_reward_total += reward
            raw_reward_total += raw_reward

            self.steps += 1
            if self.steps % 100 == 0 and self.replay_buffer.can_learn():
                batch, weights, tree_indices = self.replay_buffer.sample()
                loss, q = self._train_on_batch(*batch,weights,tree_indices)
                losses.append(loss)
                q_vals.append(q)
            if self.steps % self.update_frequency == 0 and self.replay_buffer.can_learn():
                self.soft_update_target_network()


        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        stop = self.logger.log_metrics(
                episode=episode,
                losses=losses,
                q_vals=q_vals,
                raw_reward=raw_reward_total,
                scaled_reward=scaled_reward_total,
                weights=weights,
                model=self.qnet.model
            )

        self.logger.plot_training()

        return stop


    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        state = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        return int(np.argmax(self.qnet.predict(state)))

    def soft_update_target_network(self):
        qnet_weights = self.qnet.model.get_weights()
        target_weights = self.target_net.model.get_weights()
        updated_weights = [
            self.tau * q_w + (1 - self.tau) * t_w
            for q_w, t_w in zip(qnet_weights, target_weights)
        ]
        self.target_net.model.set_weights(updated_weights)

    def _train_on_batch(self, states, actions, rewards, next_states, dones, weights, indices):
        next_q = self.target_net.model(next_states)
        targets = rewards + (self.gamma ** self.replay_buffer.n) * tf.reduce_max(next_q, axis=1) * (1 - dones)
        #print('Batch Size is',self.batch_size,'action size is',len(actions))
        action_indices = tf.stack([tf.range(self.batch_size), actions], axis=1)

        with tf.GradientTape() as tape:
            q_values = self.qnet.model(states)
            selected_q = tf.gather_nd(q_values, action_indices)
            td_errors = targets - selected_q
            # Huber loss with importance weights
            
            huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
            per_sample_loss = huber(tf.zeros_like(td_errors), td_errors)
            loss = tf.reduce_mean(per_sample_loss * weights)

        grads = tape.gradient(loss, self.qnet.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)
        self.optimiser.apply_gradients(zip(grads, self.qnet.model.trainable_variables))
    
        new_priorities = np.abs(td_errors.numpy()) + self.replay_buffer.epsilon

        self.replay_buffer.update_priorities(
            np.array(indices, dtype=int), 
            new_priorities)



        return loss.numpy(), tf.reduce_mean(selected_q).numpy()






    
        