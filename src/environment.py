import gymnasium as gym
import logging

# Valid actions for Breakout
VALID_ACTIONS = [0, 1, 2, 3]

logger = logging.getLogger(__name__)

class EpisodicLifeEnv(gym.Wrapper):
    """
    Treats the loss of a life as the end of an episode for training.
    This is a standard wrapper for Atari games to speed up learning.
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
        Steps the environment. If a life is lost, mark the episode
        as 'terminated' for learning purposes.
        """
        # The gymnasium API returns 5 values:
        # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.was_real_done = terminated or truncated

        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        
        if lives < self.lives and lives > 0:
            # A life was lost, but the game is not over.
            # Treat this as the end of an episode for learning.
            terminated = True
        
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Resets the environment. If the *real* game was over,
        reset fully. Otherwise, just press FIRE to start the new life.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Just press FIRE (action 1) to start the new life
            # We don't need the other return values
            obs, _, _, _, info = self.env.step(1)
            
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


def create_env(env_name="ALE/Breakout-v5", render_mode="rgb_array"):
    """
    Factory function to create and wrap the Atari environment.
    
    Args:
        env_name (str): The gymnasium environment ID.
        render_mode (str): The render mode for the environment.
        
    Returns:
        A fully wrapped gymnasium environment.
    """
    try:
        env = gym.make(env_name, render_mode=render_mode)
        # Apply the episodic life wrapper
        env = EpisodicLifeEnv(env)
        logger.info(f"Successfully created environment: {env_name}")
        return env
    except Exception as e:
        logger.error(f"Failed to create environment '{env_name}': {e}")
        raise

if __name__ == '__main__':
    # Simple test to ensure the environment and wrapper work
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing environment creation...")
    env = create_env()
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    obs, reward, terminated, truncated, info = env.step(1) # Fire
    logger.info(f"Observation shape after step: {obs.shape}")
    logger.info("Environment test successful.")
    env.close()
