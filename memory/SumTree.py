import numpy as np
from typing import Tuple, Any

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficiently sampling based on priority in Prioritized Experience Replay.
    """

    def __init__(self, capacity: int):
        """
        Initializes the SumTree with a fixed capacity.

        Args:
            capacity (int): Maximum number of data items that can be stored.
        """
        self.capacity = capacity  # Max number of leaf nodes (actual experiences)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # Full binary tree
        self.data = np.zeros(capacity, dtype=object)  # Experience buffer
        self.n_entries = 0  # Number of currently stored experiences
        self.data_pointer = 0  # Points to the next index for insertion

    def __len__(self) -> int:
        """
        Returns the number of stored experiences.

        Returns:
            int: Number of stored entries.
        """
        return self.n_entries

    def add(self, priority: float, data: Tuple[Any, Any, Any, Any, Any]) -> None:
        """
        Adds a new experience with a given priority to the tree.

        Args:
            priority (float): Priority score for the experience.
            data (tuple): The experience data (usually: state, action, reward, next_state, done).
        """

        # Find the index in the tree where this data will go
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # Store the data in memory
        self.update(tree_idx, priority)      # Update the tree with the new priority

        # Update counters
        self.n_entries = min(self.n_entries + 1, self.capacity)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def update(self, tree_idx: int, priority: float) -> None:
        old = self.tree[tree_idx]
        change = priority - old
        self.tree[tree_idx] = priority
        #print(f"[TREE] Updated leaf tree[{tree_idx}] from {old} to {priority} (Δ={change})")

        if change == 0.0:
            return


        # Propagate up
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            old_val = self.tree[tree_idx]
            self.tree[tree_idx] += change
            #print(f"[TREE] Propagated to tree[{tree_idx}] from {old_val} to {self.tree[tree_idx]} (Δ={change})")


    def sample_leaf(self, value: float) -> Tuple[int, float, Tuple]:
        """
        Samples a leaf node given a cumulative priority value.

        Args:
            value (float): A value in [0, total_priority), used to select a sample.

        Returns:
            Tuple[int, float, Tuple]: (index in tree, priority at index, experience data)
        """
        parent_idx = 0
        value = min(value, self.total_priority() - 1e-5)  # Clamp to avoid edge errors

        # Traverse the tree to find the appropriate leaf node
        while True:
            left = 2 * parent_idx + 1
            right = left + 1

            if left >= len(self.tree):  # If we reached a leaf
                leaf_idx = parent_idx
                break

            if value <= self.tree[left]:
                parent_idx = left
            else:
                value -= self.tree[left]
                parent_idx = right

        data_idx = leaf_idx - self.capacity + 1
        assert 0 <= data_idx < len(self.data), f"Invalid data_idx: {data_idx}"
        assert leaf_idx >= self.capacity - 1, f"Invalid leaf_idx: {leaf_idx}, not a leaf"

        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self) -> float:
        """
        Returns the total sum of priorities (i.e., the root node value).

        Returns:
            float: Total priority value.
        """
        return max(self.tree[0], 1e-8)  # Avoid zero division or sampling issues
