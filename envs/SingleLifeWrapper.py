from gym import Wrapper


class SingleLifeWrapper(Wrapper):
    """
    Gymnasium wrapper to make Atari environments end after losing a life.
    Turns multi-life games (like Demon Attack) into single-life episodes.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        """
        Reset the environment. If the previous episode ended because of a life loss,
        do a "no-op reset" (just step with a random action to continue from the current state).
        """
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives", self.lives)

        # Detect life loss
        life_lost = lives < self.lives and lives > 0
        if life_lost:
            terminated = True  # end episode on life loss

        self.lives = lives
        return obs, reward, terminated, truncated, info
