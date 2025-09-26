import numpy as np
from collections import deque
import gym
import cv2
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.SingleLifeWrapper import SingleLifeWrapper

class DemonAttackEnvironment:
    def __init__(self, render_mode=None, full_action_space=True, frameskip=4,  obs_type='grayscale'):
        """
        DemonAttack Atari environment with standard preprocessing for DQN variants:
        - Grayscale, resized to 84x84
        - Frame stacking (last 4 raw frames), channels-last: (84,84,4)
        """
        self.frame_stack = deque(maxlen=4)
        self.full_action_space = full_action_space
        self.frameskip = frameskip
        self.render_mode = render_mode if render_mode else None
        self.obs_type = obs_type

        # Create the gym environment
        self.env = gym.make(
            "DemonAttackNoFrameskip-v4",
            full_action_space=self.full_action_space,
            frameskip=self.frameskip,
            render_mode=self.render_mode,
            obs_type=self.obs_type
        )
        self.env = SingleLifeWrapper(self.env)

        # State/action dimensions
        self.state_dim = (84, 84, 4)  # channels-last
        self.action_dim = self.env.action_space.n

        # Initialize state
        self.state = self.reset()

    def preprocess_state(self, state):
        """
        Preprocess a raw RGB Atari frame:
        - Convert to grayscale
        - Resize to 84x84
        - Normalize to [0,1]
        """
        if state.ndim == 3 and state.shape[2] == 3:
            gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        else:
            gray = state  # already grayscale
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def reset(self):
        """
        Reset environment and fill frame stack with the initial frame.
        Returns:
            np.ndarray: State of shape (84, 84, 4)
        """
        obs, _ = self.env.reset()
        self.frame_stack.clear()

        preprocessed = self.preprocess_state(obs)
        for _ in range(4):
            self.frame_stack.append(preprocessed)

        # Stack along last axis for channels-last
        self.state = np.stack(self.frame_stack, axis=-1)
        return self.state

    def step(self, action):
        """
        Perform an action and update the frame stack.
        Returns:
            tuple: (state, reward, done, info)
                - state: np.ndarray (84, 84, 4)
                - reward: float
                - done: bool
                - info: dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        preprocessed = self.preprocess_state(obs)
        self.frame_stack.append(preprocessed)

        # Stack along last axis
        self.state = np.stack(self.frame_stack, axis=-1)
        return self.state, reward, done

    def render(self):
        """Render the environment."""
        self.env.render()

    def get_state(self):
        """Return the most recent stacked state."""
        return self.state

    def env_id(self):
        """Return environment ID string."""
        return self.env.spec.id
