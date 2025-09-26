# Demon Attack Atari – Prioritized Double Dueling DQN with N-Step Returns (Rainbow Implementation)

This project implements a **Rainbow-style Deep Reinforcement Learning agent** trained on the **Atari Demon Attack** environment.  
The agent combines several advanced techniques to achieve strong performance, including:

- ✅ **Double DQN** – reduces overestimation bias.  
- ✅ **Dueling Networks** – separates value and advantage streams for better generalization.  
- ✅ **Prioritized Experience Replay (PER)** – samples more informative transitions.  
- ✅ **N-Step Returns** – propagates rewards faster through time (here, `n=3`).  

The goal is to train an agent that can efficiently play **Demon Attack** using a combination of the above Rainbow components.

---

## 🧩 Agent Details

The main agent is configured as follows:

```python
Agent1 = DQNAgent(
    buffer_size=1_000_000,
    batch_size=64,
    episodes=600,
    input_shape=(84, 84, 4),
    action_size=6,
    gamma=0.999,
    epsilon=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.99,
    learning_rate=1e-4,
    tau=1,
    n=3,  # n-step returns
    optimiser='Adam',
    environment=env,
    update_frequency=10000,
    wandb_trigger=True,
    load_baseline=True,
    weights='./saved_models/best_model_ep374.weights.h5'
)
```
## ⚙️ Features

- Environment: **Atari Demon Attack** (via OpenAI Gym / ALE)  
- Neural Network: **Double Dueling Q-network (CNN-based)**  
- Replay Buffer: **Prioritized with n-step return support**  
- Exploration: **ε-greedy** with exponential decay  
- Optimizer: **Adam (1e-4)**  
- Target Network Updates: **Soft update (τ=1, i.e., hard updates) every 10,000 steps**  
- Logging: **Weights & Biases (wandb) integration**  
- Checkpointing: **Option to load pre-trained weights**  

---

## 📊 Training

1. Preprocess Atari frames into an **84×84×4 stack**.  
2. Store transitions in a **Prioritized Replay Buffer** with **3-step returns**.  
3. Sample **mini-batches of size 64** for training.  
4. Compute **Double DQN targets** with **dueling heads**.  
5. Periodically update the **target network**.  
6. Log metrics and training progress to **wandb**.  

## 📈 Results

- Using **n-step returns** improved the consistency of higher scores.  
- After training the agent for **1000 episodes** across **3 independent runs (checkpoints)**, the agent achieved a score of **450 on a single life**.  
- Further optimization is possible — many hyperparameters remain to be tuned, and additional improvements could be made.  

### 🚀 Future Work

Planned extensions and experiments include:

- Distributional Q-learning (**C51 / QR-DQN**)  
- **Noisy Nets** for exploration  
- **Full Rainbow integration** (all 6 components)  
- **Parallel training** (Ape-X style)  
- Variations on DQN with **self-attention mechanisms**  
- Exploring alternative **state representations** (beyond raw frames, which can be sparse)  

I will continue iterating on this project after the **Kaggle Market Prediction** competition (to be released in a separate repository). Wish me luck! 🍀  

---

## 🙌 Acknowledgements

- [Rainbow DQN: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)  
- [OpenAI Gym Atari environments](https://gymnasium.farama.org/)  
- DeepMind’s original **DQN** and **Rainbow** research papers  
