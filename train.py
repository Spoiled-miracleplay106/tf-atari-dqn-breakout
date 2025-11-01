import tensorflow.compat.v1 as tf
import os
import argparse
import logging
import sys

from src.environment import create_env
from src.preprocessing import StateProcessor
from src.model import Estimator
from src.trainer import deep_q_learning
from src.utils import setup_logging

# Disable TF2 behavior and suppress warnings
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on Atari Breakout."
    )
    
    # Environment and Directories
    parser.add_argument(
        "--env_name", 
        type=str, 
        default="ALE/Breakout-v5",
        help="The gymnasium environment ID."
    )
    parser.add_argument(
        "--experiment_dir", 
        type=str, 
        default="./experiments/breakout_v5",
        help="Directory to save checkpoints, summaries, and videos."
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs",
        help="Directory to save log files."
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )

    # Training Hyperparameters
    parser.add_argument(
        "--num_episodes", 
        type=int, 
        default=10000,
        help="Number of episodes to train."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Size of the minibatch to sample from the replay buffer."
    )
    parser.add_argument(
        "--replay_memory_size", 
        type=int, 
        default=500000,
        help="Size of the replay buffer."
    )
    parser.add_argument(
        "--replay_memory_init_size", 
        type=int, 
        default=50000,
        help="Number of random transitions to populate the buffer with."
    )
    parser.add_argument(
        "--update_target_estimator_every", 
        type=int, 
        default=10000,
        help="Steps between copying Q-Network weights to Target-Network."
    )
    parser.add_argument(
        "--discount_factor", 
        type=float, 
        default=0.99,
        help="Discount factor (gamma) for future rewards."
    )
    parser.add_argument(
        "--epsilon_start", 
        type=float, 
        default=1.0,
        help="Starting value for epsilon in epsilon-greedy policy."
    )
    parser.add_argument(
        "--epsilon_end", 
        type=float, 
        default=0.1,
        help="Final value for epsilon."
    )
    parser.add_argument(
        "--epsilon_decay_steps", 
        type=int, 
        default=500000,
        help="Number of steps to decay epsilon over."
    )
    parser.add_argument(
        "--max_steps_per_episode", 
        type=int, 
        default=10000,
        help="Maximum steps per episode before truncating."
    )
    parser.add_argument(
        "--record_video_every", 
        type=int, 
        default=50,
        help="Record a video every N episodes."
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # --- Setup Logging ---
    try:
        setup_logging(log_level=args.log_level, log_dir=args.log_dir)
        logger = logging.getLogger(__name__)
        logger.info("Starting DQN Training Script")
        logger.info(f"Arguments: {vars(args)}")
    except Exception as e:
        print(f"Error setting up logging: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Initialize Environment, Model, and Processor ---
    try:
        env = create_env(args.env_name)
        
        tf.reset_default_graph()
        
        # Create global step variable
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Create estimators
        q_estimator = Estimator(
            scope="q_estimator", 
            summaries_dir=args.experiment_dir
        )
        target_estimator = Estimator(scope="target_q")
        
        # State processor
        state_processor = StateProcessor()
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    # --- Run Training ---
    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Avoid TF allocating all GPU mem
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            
            logger.info("TensorFlow session started and variables initialized.")
            
            for t, stats, last_loss in deep_q_learning(
                sess,
                env,
                q_estimator=q_estimator,
                target_estimator=target_estimator,
                state_processor=state_processor,
                num_episodes=args.num_episodes,
                experiment_dir=args.experiment_dir,
                replay_memory_size=args.replay_memory_size,
                replay_memory_init_size=args.replay_memory_init_size,
                update_target_estimator_every=args.update_target_estimator_every,
                discount_factor=args.discount_factor,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                batch_size=args.batch_size,
                record_video_every=args.record_video_every,
                max_steps_per_episode=args.max_steps_per_episode
            ):
                current_episode = len(stats.episode_rewards)
                reward = stats.episode_rewards[-1]
                length = stats.episode_lengths[-1]
                
                # Log progress
                log_msg = (
                    f"Episode {current_episode}/{args.num_episodes} | "
                    f"Total Steps: {t} | "
                    f"Reward: {reward:.2f} | "
                    f"Length: {length} | "
                    f"Last Loss: {last_loss:.6f}"
                )
                logger.info(log_msg)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    except Exception as e:
        logger.error(f"An unhandled error occurred during training: {e}", exc_info=True)
    finally:
        env.close()
        logger.info("Environment closed. Training script finished.")

if __name__ == "__main__":
    main()
