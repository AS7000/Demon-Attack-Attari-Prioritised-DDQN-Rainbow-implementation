import numpy as np
from typing import Any, List, Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from memory.SumTree import SumTree

class PrioritizedReplayBuffer:
    """
    A memory buffer that stores experiences and samples them based on priority.
    Prioritized Experience Replay improves learning efficiency by replaying important transitions more frequently.
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        epsilon: float = 1e-6
    ):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of transitions to store.
            batch_size (int): Number of experiences to sample during training.
            alpha (float): Priority exponent; 0 = uniform sampling, 1 = full prioritization.
            beta (float): Initial importance-sampling weight exponent.
            beta_increment (float): Increment per sampling step (towards 1).
            epsilon (float): Small value to avoid zero priority.
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.max_priority = 1.0  # track max priority
        self.frame = 1           # count calls to sample() for beta annealing


    def __len__(self) -> int:
        """Returns the number of stored experiences."""
        return len(self.tree.data)

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        Adds a new experience to the buffer with a default priority.
        """
        data = (state, action, reward, next_state, done)

        if not isinstance(data, tuple) or len(data) != 5:
            raise ValueError(f"Invalid experience format: {data}")

        # use the current max priority for new samples
        priority = (self.max_priority + self.epsilon) ** self.alpha
        self.tree.add(priority, data)


    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a batch of experiences using stratified proportional prioritization.
        """
        batch: List[Tuple] = []
        idxs: List[int] = []
        priorities: List[float] = []

        total_priority = self.tree.total_priority()
        if total_priority <= 0.0:
            raise ValueError("Cannot sample: total priority is zero or negative.")

        # anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.frame += 1

        # stratified sampling
        segment = total_priority / self.batch_size
        for i in range(self.batch_size):
            while True:  # keep retrying until valid sample found
                s = np.random.uniform(i * segment, (i + 1) * segment)
                idx, priority, data = self.tree.sample_leaf(s)

                if isinstance(data, tuple) and len(data) == 5:
                    batch.append(data)
                    idxs.append(idx)
                    priorities.append(priority)
                    break
        if len(batch) == 0:
            raise RuntimeError("No valid samples found in buffer. Check your SumTree implementation.")

        priorities = np.array(priorities, dtype=np.float32)
        probs = priorities / (total_priority + self.epsilon)
        assert np.all(probs <= 1.0), "Invalid probabilities"
        probs = np.clip(probs, 1e-8, 1.0)  # avoid divide-by-zero
        
        # importance sampling
        weights = (len(self.tree.data) * probs) ** (-self.beta)
        weights /= (weights.max() + 1e-8)  # normalize safely
        weights = weights.astype(np.float32)

        assert weights.min() >= 0
        assert weights.max() < 10, f"Weights exploding: {weights.max()}"


        # transpose batch
        batch = list(zip(*batch))
        return (
            np.stack(batch[0]),  # states
            np.array(batch[1]),  # actions
            np.array(batch[2]),  # rewards
            np.stack(batch[3]),  # next_states
            np.array(batch[4]),  # dones
            np.array(idxs),      # indices
            weights              # importance-sampling weights
        )



    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Updates the priorities of sampled transitions.
        """

        for idx, new_priority in zip(indices, priorities):
            adjusted = (abs(new_priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, adjusted)
            self.max_priority = max(self.max_priority, adjusted)


    def can_learn(self) -> bool:
        """
        Checks if buffer has enough samples to start training.
        """
        return (
            self.tree.n_entries >= (0.1 * self.tree.capacity) and
            len(self.tree.data) >= self.batch_size
        )

