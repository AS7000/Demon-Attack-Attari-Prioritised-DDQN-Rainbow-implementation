import numpy as np
'''
def reward_shaping(raw_reward, done, normalizer=None):
    if done:
        raw_reward -= 1.0   

    if normalizer:
        raw_reward = normalizer.normalize(raw_reward)
    return raw_reward
'''
def reward_shaping(raw_reward):
    """
    Atari-style reward clipping: reward = sign(raw_reward).
    Returns -1, 0, or +1.
    """
    return np.sign(raw_reward)

