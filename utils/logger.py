import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import wandb
import os
from utils.EarlyStopping import EarlyStopping

class Logger:
    def __init__(self, config, env, model, save_path="saved_models", use_wandb=False):
        self.episode_rewards = []
        self.mean_losses = []
        self.mean_exp_return = []
        self.q_eval_track = []
        self.raw_rewards = []
        self.scaled_rewards = []
        self.fixed_states = []
        self.weight_stats = []
        self.env = env
        self.model = model
        self.early_stopper = EarlyStopping(20)

        self.best_score = -np.inf

        self.use_wandb = use_wandb
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.config = config
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.collect_fixed_states()

        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        wandb.init(project=self.config.get("project_name", "RL_Project"), config=self.config)

    def log_metrics(self, episode, losses, q_vals,
                    raw_reward, scaled_reward,weights, model):

        mean_loss = np.mean(losses) if losses else 0
        mean_q_val = np.mean(q_vals) if q_vals else 0
        avg_q_fixed_states = self.evaluate_q_on_fixed_states()
        w_min, w_max, w_mean = ((np.min(weights), np.max(weights), np.mean(weights)) if len(self.weight_stats) > 0 else (0, 0, 0))


        # Append metrics
        self.mean_losses.append(mean_loss)
        self.mean_exp_return.append(mean_q_val)
        self.q_eval_track.append(avg_q_fixed_states)
        self.raw_rewards.append(raw_reward)
        self.scaled_rewards.append(scaled_reward)
        self.weight_stats.append((w_min, w_max, w_mean))

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.use_wandb:
            wandb.log({
                "episode": episode,
                "mean_loss": mean_loss,
                "mean_q_value": mean_q_val,
                "avg_max_q_on_fixed_states": avg_q_fixed_states,
                "raw_reward": raw_reward,
                "scaled_reward": scaled_reward,
                "epsilon": self.epsilon,
                "weights/min": w_min,
                "weights/max": w_max,
                "weights/mean": w_mean
            })

        # Save model if improved
        window = 10  # moving average window
        avg_reward = np.mean(self.raw_rewards[-window:]) if len(self.raw_rewards) >= window else 0

        if avg_reward > self.best_score:
            print(f"[Model Saved] Avg Reward: {avg_reward:.2f} | Mean Q: {mean_q_val:.2f} | Avg Fixed Q: {avg_q_fixed_states:.2f}")
            save_file = os.path.join(self.save_path, f"best_model_ep{episode}.weights.h5")
            model.save_weights(save_file)
            if self.use_wandb:
                wandb.save(save_file)
            self.best_score = avg_reward

        if self.early_stopper.step(avg_reward) and self.early_stopper.should_stop:
            print(f"Early stopping triggered at episode {episode} | Best reward: {self.early_stopper.best_score:.2f}")
            return True  # <-- signal to trainer to stop

        clear_output(wait=True)
        print(f"Episode {episode} | Loss: {mean_loss:.4f} | "
              f"Q: {mean_q_val:.4f} | Avg Fixed Q: {avg_q_fixed_states:.4f} | "
              f"Epsilon: {self.epsilon:.2f} | Raw: {raw_reward:.1f}, Scaled: {scaled_reward:.1f}")
        print('Average Reward for the last 10 episodes:', avg_reward, '- Best Overall:',self.best_score)
        

        
        return False # keep training


    def plot_training(self):
        fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        axs[0, 0].plot(self.mean_losses, label="Mean Loss", color='orange')
        axs[0, 1].plot(self.mean_exp_return, label="Mean Q-Value", color='green')
        axs[1, 0].plot(self.q_eval_track, label="Avg Max Q (Fixed States)", color='purple')
        axs[1, 1].plot(self.raw_rewards, label="Raw Reward", linestyle='--', color='red')
        axs[1, 1].plot(self.scaled_rewards, label="Scaled Reward", linestyle='-', color='blue')

        titles = ["Mean Loss", "Mean Q-Value", "Diagnostic Q", "Raw vs Scaled"]
        for ax, title in zip(axs.flat, titles):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_test(self):
        clear_output(wait=True)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(self.episode_rewards, label="Test Reward")
        axs[1].plot(self.q_eval_track, label="Avg Max Q (Fixed States)", color='purple')
        titles = ["Test Rewards", "Diagnostic Q"]
        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.show()


    def collect_fixed_states(self, num_states=1000):
        self.fixed_states.clear()
        obs = self.env.reset()
        for _ in range(num_states):
            self.fixed_states.append(obs)
            action = self.env.env.action_space.sample()
            obs, _, done = self.env.step(action)
            if done:
                obs = self.env.reset()

    def evaluate_q_on_fixed_states(self):
        if not self.fixed_states:
            return 0  # No states to evaluate

        values = []
        for state in self.fixed_states:
            state_array = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
            q_vals = self.model.predict(state_array, verbose=0)
            max_q = np.max(q_vals)
            values.append(max_q)
        return np.mean(values)
    
    def close(self):
        wandb.finish()


