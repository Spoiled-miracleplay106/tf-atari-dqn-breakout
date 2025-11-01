# Deep Q-Learning for Atari Breakout

This project is a TensorFlow and Gymnasium implementation of the Deep Q-Learning (DQN) algorithm, developed for the M.S. Deep Learning course. It trains an agent to play the Atari game 'Breakout' directly from pixel inputs, based on the principles from the "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013/2015) paper.

## Features

* **Modular Codebase:** All logic is separated into components (`model`, `agent`, `trainer`, `environment`).
* **Efficient Experience Replay:** Uses `collections.deque` for a highly efficient $O(1)$ replay buffer.
* **$\epsilon$-Greedy Policy:** Implements an epsilon-greedy exploration strategy with a linear decay schedule.
* **Stable Target Network:** Uses a separate target network (a core DQN technique) for stabilized Q-value estimation.
* **Standard Preprocessing:** Implements the standard Atari preprocessing pipeline (Grayscale, Resize, Frame Stacking).
* **Logging & Monitoring:** Includes comprehensive logging (to console and file) and TensorBoard integration.
* **Configurable Training:** All hyperparameters are exposed as command-line arguments via `train.py`.

## Core Concepts & Techniques

* Reinforcement Learning (RL)
* Q-Learning & The Bellman Equation
* Deep Q-Networks (DQN)
* Experience Replay
* Target Networks
* Convolutional Neural Networks (CNNs)

---

## How It Works

This project trains an agent to master Breakout by learning a policy that maps game states (pixels) to optimal actions (joystick movements). It does this using a Deep Q-Network (DQN).

### 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a paradigm where an **agent** learns to interact with an **environment** to maximize a cumulative **reward**. The agent observes a **state** ($S$), takes an **action** ($A$), and receives a **reward** ($R$) and a new state ($S'$) from the environment. The agent's "brain" is its **policy** ($\pi$), which dictates what action to take in any given state.

### 2. Q-Learning and the Bellman Equation

The goal is to find the *optimal* policy. Q-Learning does this by learning a **Q-function**, $Q(s, a)$.

* **Q-Function $Q(s, a)$:** This function represents the "quality" of taking action $a$ in state $s$. It's defined as the total expected future reward the agent can get if it starts in state $s$, takes action $a$, and acts optimally thereafter.

* **The Bellman Equation:** The optimal Q-function, $Q^*(s, a)$, must satisfy the Bellman optimality equation. This equation breaks down the value of an action into two parts: the immediate reward ($R$) and the discounted value of the *best* action it can take in the *next* state ($s'$).

  $$Q^\*(s, a) = \mathbb{E}\_{s'}[R + \gamma \max_{a'} Q^*(s', a') | s, a]$$
  
    * $R$ is the immediate reward.
    * $\gamma$ (gamma) is the *discount factor* (e.g., 0.99), which makes future rewards slightly less valuable than immediate ones.
    * $\max_{a'} Q^*(s', a')$ is the *best possible Q-value* in the next state.

In simple environments (like a small grid), we can store these Q-values in a table. In Q-Learning, we iteratively update this table using the **Temporal Difference (TD) update rule**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [ (R + \gamma \max_{a'} Q(s', a')) - Q(s, a) ]$$

* $\alpha$ (alpha) is the *learning rate*.
* The term in the brackets, $[...]$, is the **TD-Error**: the difference between our *target* value ( $R + \gamma \max_{a'} Q(s', a')$ ) and our *current prediction* ( $Q(s, a)$ ).

### 3. Deep Q-Networks (DQN)

In Atari, the state space is massive (a $210 \times 160 \times 3$ image). A Q-table is impossible.

The solution is to use a deep neural network (a CNN, in this case) to *approximate* the Q-function: $Q(s, a; \theta) \approx Q^*(s, a)$, where $\theta$ represents the network's weights.

* **Input:** The state $s$ (a stack of 4 processed $84 \times 84$ frames).
* **Output:** A vector of Q-values, one for each possible action (e.g., `[Q(s, 'noop'), Q(s, 'fire'), Q(s, 'left'), Q(s, 'right')]`).
* **Loss Function:** We can treat this as a regression problem. Our goal is to make our network's prediction, $Q(s, a; \theta)$, as close as possible to the Bellman target, $y_i = R + \gamma \max_{a'} Q(s', a'; \theta)$.

This leads to a simple Mean Squared Error (MSE) loss:
$$L(\theta) = \mathbb{E}[(y_i - Q(s, a; \theta))^2]$$

However, training this naively is highly unstable. DQN introduces two key techniques to solve this.

### 4. Core Components Explained

#### 1. Experience Replay

* **Problem:** Training a NN with consecutive, highly-correlated states (e.g., frames in a game) violates the IID (Independent and Identically Distributed) data assumption. This makes training very unstable.
* **Solution:** We create a large **Replay Buffer** (in our case, `src/agent.py:ReplayBuffer`).
    1.  As the agent plays, it stores transitions $(s, a, r, s')$ in this buffer.
    2.  During training, instead of using the *last* transition, we sample a random mini-batch of transitions from this buffer.
* **Benefits:** This breaks the correlations between consecutive samples and smooths out the training process, leading to much greater stability.

#### 2. Target Network

* **Problem:** In our loss function, $L(\theta) = \mathbb{E}[(y_i - Q(s, a; \theta))^2]$, the *target* $y_i = R + \gamma \max_{a'} Q(s', a'; \theta)$ is calculated using the *same weights* $\theta$ that we are actively trying to update.
* This is like "chasing a moving target." As we update $\theta$ to match $y_i$, the value of $y_i$ itself changes. This is another major source of instability.
* **Solution:** We use **two** neural networks.
    1.  **Q-Network ($\theta$):** This is the main network we update on every training step. It's used to calculate $Q(s, a; \theta)$.
    2.  **Target-Network ($\theta^-$):** This is a clone of the Q-Network. Its weights are *frozen* for many steps (e.g., 10,000). It is only used to calculate the target: $y_i = R + \gamma \max_{a'} Q(s', a'; \theta^-)$.
* **Benefits:** This provides a stable, consistent target for the Q-Network to regress towards. Every 10,000 steps, we copy the weights from the Q-Network to the Target-Network (`src/agent.py:ModelParametersCopier`).

### 5. Project Architecture

* `train.py`: The main executable script. Parses arguments, sets up logging, and starts the training process.
* `src/trainer.py`: Contains the main `deep_q_learning` training loop. This function orchestrates the entire process: interacting with the environment, sampling from the replay buffer, calculating targets, and updating the Q-network.
* `src/model.py`: Defines the `Estimator` class, which builds the CNN architecture (3 conv layers, 2 dense layers) as specified in the paper using TensorFlow 1.x.
* `src/agent.py`: Contains the `ReplayBuffer` (using `deque`), the `ModelParametersCopier` (for updating the target network), and the `make_epsilon_greedy_policy` function.
* `src/environment.py`: Provides the `EpisodicLifeEnv` wrapper (which treats a lost life as an episode's end to speed up learning) and a `create_env` factory function.
* `src/preprocessing.py`: Contains the `StateProcessor` class, which uses a TF graph to efficiently convert raw $210 \times 160 \times 3$ RGB frames into processed $84 \times 84 \times 1$ grayscale images.
* `src/utils.py`: Contains logging setup (`setup_logging`) and helper structures.

---

## Project Structure

```
tf-atari-dqn-breakout/
├── .gitignore              # Ignores Python caches, logs, and experiment data
├── LICENSE                 # MIT License
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── train.py                # Main script to run training
├── run_training.ipynb      # Notebook to explain and run train.py
├── logs/                   # Directory for .log files
│   └── .gitkeep
├── experiments/            # Directory for TensorBoard summaries, checkpoints, videos
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── agent.py            # ReplayBuffer, Policy, and Target-Network Copier
    ├── environment.py      # Environment wrappers (EpisodicLifeEnv)
    ├── model.py            # Estimator (DQN) class
    ├── preprocessing.py    # StateProcessor (pixel processing) class
    ├── trainer.py          # The main deep_q_learning training loop
    └── utils.py            # Logging setup and helper classes
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/tf-atari-dqn-breakout.git
    cd tf-atari-dqn-breakout
    ```

2.  **Set up Environment and Dependencies:**
    It's highly recommended to use a Python virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    # Install gymnasium with Atari support AND accept the ROM license
    pip install "gymnasium[atari,accept-rom-license]"
    ```

3.  **Run Training:**
    You can run the training with default parameters using the main script. Logs will appear in the console and be saved to `logs/training.log`. TensorBoard summaries will be saved in `experiments/breakout_v5`.
    ```bash
    python train.py
    ```

4.  **Example Usage (Custom Parameters):**
    You can override any hyperparameter via command-line arguments. For example, to run a shorter training session:
    ```bash
    python train.py \
        --num_episodes 1000 \
        --replay_memory_init_size 10000 \
        --epsilon_decay_steps 100000 \
        --experiment_dir ./experiments/test_run
    ```
    Run `python train.py --help` to see all available options.

5.  **Monitor with TensorBoard:**
    While the model is training, you can launch TensorBoard to monitor the agent's rewards, loss, and other metrics in real-time.
    ```bash
    # Run this in a separate terminal
    tensorboard --logdir=./experiments
    ```
    Then open your browser to `http://localhost:6006`.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
