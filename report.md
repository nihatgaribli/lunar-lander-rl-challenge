# LunarLander-v3 RL Final Project Report

## 1. Track Choice and Algorithms
**Primary Track**: Track V (Value-based) using **Deep Q-Network (DQN)**.
**Bonus Track**: Track P (Policy-based) using **Proximal Policy Optimization (PPO)**.

DQN was chosen as the primary track because the continuous state space and discrete action space of LunarLander are perfectly suited for value-based methods with experience replay. DQN is highly sample-efficient and stable once the target network and replay buffer are properly tuned, making it a reliable choice for the competitive leaderboard. PPO was implemented as a bonus to explore policy gradient methods, utilizing a clipped surrogate objective and Generalized Advantage Estimation (GAE).

## 2. Hyperparameters and Tuning

### DQN Hyperparameters (Primary)
- **Architecture**: 3-layer MLP (8 → 128 → 128 → 4) with ReLU activations. A slightly larger network (128 units vs standard 64) was chosen to reliably capture the dynamics of the lander without overfitting.
- **Learning Rate**: `5e-4` (Adam). Found via tuning; `1e-3` was too aggressive and caused instability, while `1e-4` learned too slowly.
- **Gamma ($\gamma$)**: `0.99`. Standard for episodic tasks.
- **Target Network Update ($\tau$)**: `0.005` (soft update). Provided much smoother convergence than hard updates every $N$ steps.
- **Exploration ($\epsilon$)**: Linear decay from `1.0` to `0.01` over the first `80,000` steps. This ensures sufficient exploration in the early stages and transitions to greedy exploitation before the 150K step budget is exhausted.
- **Buffer Capacity**: `100,000`. Batch Size: `64`.

### PPO Hyperparameters (Bonus)
- **Architecture**: Separate actor and critic networks (8 → 64 → 64). Orthogonal initialization was applied to both, with the actor output layer scaled by `0.01` and the critic by `1.0`.
- **Learning Rate**: `2.5e-4` (Adam) with linear annealing down to 0 over the course of training.
- **Gamma ($\gamma$)**: `0.99`, **GAE Lambda ($\lambda$)**: `0.95`.
- **Clip Epsilon**: `0.2`, **Entropy Coefficient**: `0.01`.
- **Rollout Steps**: `2048` per update, optimized over `4` epochs with minibatches of size `64`.

## 3. Learning Curves and Seed Variance
Both agents were trained using 3 different random seeds for network initialization and environment exploration. 

### DQN Results
- **Seed 42**: Solved the environment smoothly. Public seed evaluation: **231.2 ± 43.1**.
- **Seed 123**: Solved the environment. Public seed evaluation: **229.1 ± 62.4**.
- **Seed 456**: Failed to consistently solve the environment, converging to a sub-optimal policy. Public seed evaluation: **144.5 ± 92.6**.

*Mean across 3 seeds*: **201.6 ± 40.4**
*Best Agent Submitted*: **Seed 42**

### PPO Results
- **Seed 42**: Public seed evaluation: **-551.6 ± 57.5**.
- **Seed 123**: Public seed evaluation: **-352.0 ± 30.0**.
- **Seed 456**: Public seed evaluation: **-628.6 ± 61.3**.

*Mean across 3 seeds*: **-510.7 ± 116.6**
*Best Agent Submitted*: **Seed 123**

**Discussion on PPO vs DQN:**
The results clearly show that PPO struggled to solve the environment within the strict 150,000 step limit, whereas DQN solved it successfully. This is an expected outcome: PPO is an on-policy algorithm that requires fresh data for each update, making it less sample-efficient than off-policy algorithms like DQN which reuse past experiences via a replay buffer. While PPO is highly stable, it typically requires 500K - 1M steps to solve LunarLander. The 150K step budget constraint strongly favors sample-efficient off-policy methods (DQN).

**Discussion on Seed Variance:**
The performance spread across seeds in DQN (especially the failure of Seed 456) highlights the inherent instability of deep reinforcement learning. Despite identical hyperparameters and code, the random initialization of the network weights and the stochasticity of the $\epsilon$-greedy exploration trajectory caused one seed to get stuck in a local optimum (likely hovering or landing sub-optimally). This emphasizes the importance of evaluating over multiple seeds rather than relying on a "lucky" run.

## 4. Compute Budget
- DQN Step Count: `450,000` steps (150,000 per seed × 3 seeds)
- PPO Step Count: `450,000` steps (150,000 per seed × 3 seeds)
- **Total Environment Steps**: `900,000` total steps used across both tracks. 

The primary submission (DQN) strictly utilized only `450,000` steps, which is well within the 500,000 budget constraint. The PPO bonus track utilized an additional 450,000 steps.
