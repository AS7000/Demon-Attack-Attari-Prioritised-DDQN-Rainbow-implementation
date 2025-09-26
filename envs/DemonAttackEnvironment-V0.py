import numpy as np
from collections import deque
import gym
import cv2



class DemonAttackEnvironment:
    def __init__(self, render_mode=None, full_action_space=True, frameskip=3, obs_type='grayscale'):
        """
        Initializes the DemonAttack Atari environment with preprocessing and frame stacking.

        Args:
            render_mode (str or None): Rendering mode (e.g., "human" or None).
        """
        self.frame_stack = deque(maxlen=4)  # To hold the last 4 preprocessed frames
        self.full_action_space = full_action_space
        self.frameskip = frameskip
        self.render_mode = render_mode if render_mode else None
        self.obs_type = obs_type


        # Create the gym environment with deterministic frame skipping and full action space
        self.env = gym.make("DemonAttackNoFrameskip-v4",
                            full_action_space=self.full_action_space,
                            frameskip=self.frameskip,
                            render_mode=self.render_mode,
                            obs_type=self.obs_type)


        self.state_dim = (4, 84, 84)             # Shape: (stacked_frames, height, width)
        self.action_dim = self.env.action_space.n  # Number of possible actions

        # Reset environment to initialize the state
        self.state = self.reset()

    def preprocess_state(self, state):
        """
        Converts RGB state to grayscale, resizes it, and normalizes pixel values.

        Args:
            state (np.ndarray): The raw RGB frame from the environment.

        Returns:
            np.ndarray: Preprocessed grayscale frame of shape (84, 84).
        """
        if isinstance(state, np.ndarray) and state.ndim == 3 and state.shape[2] == 3:
            gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA) / 255.0    # Resize and normalize
            return np.float32(resized)                      # Ensure float32 type
        elif isinstance(state, np.ndarray) and state.ndim == 2:
            resized = cv2.resize(state, (84, 84),interpolation=cv2.INTER_AREA) / 255.0    # Resize and normalize
            return np.float32(resized)                      # Ensure float32 type
        else:
            raise ValueError(
                f"Unsupported state type: {type(state)} with shape {getattr(state, 'shape', None)}"
            )

    def reset(self):
        """
        Resets the environment and initializes the frame stack.

        Returns:
            np.ndarray: Initial stacked state of shape (84, 84, 4).
        """
        self.state, _ = self.env.reset()
        self.frame_stack.clear()

        # Preprocess the initial frame
        preprocessed = self.preprocess_state(self.state)

        # Stack 4 identical frames to initialize the state
        for _ in range(4):
            self.frame_stack.append(preprocessed)

        return np.stack(self.frame_stack, axis=2)

    def step(self, action):
        """
        Performs an action in the environment.

        Args:
            action (int): Action index to perform.

        Returns:
            tuple: (stacked_state, reward, done, lives)
                - stacked_state (np.ndarray): Updated frame stack.
                - reward (float): Reward from the environment.
                - done (bool): Whether the episode has ended.
                - lives (int): Number of lives left (from info dict).
        """
        self.state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Preprocess and update frame stack
        preprocessed = self.preprocess_state(self.state)
        self.frame_stack.append(preprocessed)

        return np.stack(self.frame_stack, axis=2), reward, done, info['lives']

    def render(self):
        """
        Renders the environment (if render_mode is set to "human").
        """
        self.env.render()

    def get_state(self):
        """
        Returns the most recent raw state frame (not the stacked one).
        """
        return self.state

    def env_id(self):
        """
        Returns the environment ID string.
        """
        return self.env.spec.id

