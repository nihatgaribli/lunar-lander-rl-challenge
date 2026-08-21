# -*- coding: utf-8 -*-
"""
Frozen Lake -- 4x5 Gridworld, Assignments 1-4 (Summer Semester 2026)

Name: Nihat Garibli
ID:   5425819

Environment (as specified in the assignment)
--------------------------------------------
* 4x5 grid, holes (H) and the goal (G) are terminal states.
* Every transition gives a step reward of -0.08.
  In addition: +1 when entering the goal, -1 when falling into a hole
  (so a transition into G is worth 0.92 and into H is -1.08).
* Actions: 0 = W (left), 1 = S (down), 2 = E (right), 3 = N (up).
* Slippery surface: intended direction w.p. 0.8, each perpendicular
  direction w.p. 0.1  (slip_prob = 0.2, split evenly).
* Rigid walls: bumping into a wall keeps the agent in place.
* Discount factor gamma = 0.95.
* Every episode starts from a uniformly random non-terminal state.

Algorithms
----------
1. Dynamic Programming : Policy Iteration + Value Iteration (model based, uses env.P)
2. Monte Carlo         : GLIE, first-visit, constant step size alpha
3. Sarsa               : on-policy TD(0) control
4. Q-Learning          : off-policy TD(0) control
"""

import os
import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

import matplotlib

# Figures are saved to ./plots ; set RL_SHOW_PLOTS=0 to run head-less (no GUI).
SHOW_PLOTS = os.environ.get("RL_SHOW_PLOTS", "1") == "1"
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

try:  # the unicode arrows must survive the Windows console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Global constants, taken directly from the assignment sheet
# --------------------------------------------------------------------------- #
GAMMA = 0.95
STEP_REWARD = -0.08
GOAL_REWARD = 1.0
HOLE_REWARD = -1.0
SLIP_PROB = 0.2

ACTION_ARROWS = ["←", "↓", "→", "↑"]  # left, down, right, up
ACTION_NAMES = ["W (left)", "S (down)", "E (right)", "N (up)"]

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Statistics filled in by the algorithms, used for the final comparison table.
RUN_STATS = {}

# Module level RNG so that a whole run is reproducible via set_seed().
RNG = np.random.default_rng(0)


def set_seed(seed):
    """Seed the module RNG (used by the epsilon-greedy behaviour policy)."""
    global RNG
    RNG = np.random.default_rng(seed)


# =========================================================================== #
#                             THE ENVIRONMENT                                 #
# =========================================================================== #
class CustomFrozenLakeEnv(gym.Env):
    """A customised 4x5 FrozenLake that follows the gymnasium Env API.

    The full model (transition probabilities and rewards) is exposed through
    ``self.P[s][a] = [(prob, next_state, reward, terminated), ...]`` which is
    exactly what the Dynamic Programming algorithms of Assignment 1 need.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        map_name="4x5",
        is_slippery=True,
        slip_prob=SLIP_PROB,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        hole_reward=HOLE_REWARD,
        random_start=True,
        render_mode=None,
    ):
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.hole_reward = hole_reward
        self.random_start = random_start
        self.render_mode = render_mode

        #  S F F F F
        #  F F F H F
        #  F F F F F
        #  H F F F G
        self.desc = np.asarray(["SFFFF", "FFFHF", "FFFFF", "HFFFG"], dtype="c")

        self.nrow, self.ncol = self.desc.shape
        self.nA = 4
        self.nS = self.nrow * self.ncol

        flat = self.desc.reshape(-1)
        self.terminal_states = np.array([bytes(c) in b"GH" for c in flat])
        self.hole_states = np.array([bytes(c) == b"H" for c in flat])
        self.goal_states = np.array([bytes(c) == b"G" for c in flat])
        # Episodes start in any non-terminal square (assignment: random start state).
        self.start_states = np.flatnonzero(~self.terminal_states)

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self._init_transitions()
        self._build_sampling_tables()

        self.observation_space = Discrete(self.nS)
        self.action_space = Discrete(self.nA)

        self.s = int(self.start_states[0])

    # ------------------------------------------------------------------ #
    # geometry helpers
    # ------------------------------------------------------------------ #
    def to_s(self, row, col):
        return row * self.ncol + col

    def inc(self, row, col, a):
        """Move one square in direction ``a``; the walls are rigid."""
        if a == 0:      # left  (W)
            col = max(col - 1, 0)
        elif a == 1:    # down  (S)
            row = min(row + 1, self.nrow - 1)
        elif a == 2:    # right (E)
            col = min(col + 1, self.ncol - 1)
        elif a == 3:    # up    (N)
            row = max(row - 1, 0)
        return (row, col)

    def _reward(self, newletter):
        """Step cost on every transition, plus the terminal bonus / penalty."""
        r = self.step_reward
        if newletter == b"G":
            r += self.goal_reward
        elif newletter == b"H":
            r += self.hole_reward
        return float(r)

    # ------------------------------------------------------------------ #
    # the model  P[s][a] -> [(prob, s', r, done), ...]
    # ------------------------------------------------------------------ #
    def _init_transitions(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self.to_s(row, col)
                letter = self.desc[row, col]
                for a in range(self.nA):
                    li = self.P[s][a]
                    if bytes(letter) in b"GH":
                        # Absorbing terminal state: no further reward.
                        li.append((1.0, s, 0.0, True))
                        continue

                    if self.is_slippery:
                        # (a-1)%4 and (a+1)%4 are the two perpendicular
                        # directions  =>  0.8 / 0.1 / 0.1 as required.
                        moves = [((a - 1) % 4, self.slip_prob / 2),
                                 (a, 1.0 - self.slip_prob),
                                 ((a + 1) % 4, self.slip_prob / 2)]
                    else:
                        moves = [(a, 1.0)]

                    for b, prob in moves:
                        newrow, newcol = self.inc(row, col, b)
                        newstate = self.to_s(newrow, newcol)
                        newletter = self.desc[newrow, newcol]
                        done = bytes(newletter) in b"GH"
                        li.append((prob, newstate, self._reward(newletter), done))

    def _build_sampling_tables(self):
        """Pre-compute cumulative probabilities so that step() stays cheap."""
        self._cum, self._nxt, self._rew, self._done = {}, {}, {}, {}
        for s in range(self.nS):
            for a in range(self.nA):
                tr = self.P[s][a]
                self._cum[(s, a)] = np.cumsum([t[0] for t in tr])
                self._nxt[(s, a)] = np.array([t[1] for t in tr], dtype=np.int64)
                self._rew[(s, a)] = np.array([t[2] for t in tr], dtype=np.float64)
                self._done[(s, a)] = np.array([t[3] for t in tr], dtype=bool)

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "state" in options:
            self.s = int(options["state"])
        elif self.random_start:
            self.s = int(self.np_random.choice(self.start_states))
        else:
            self.s = 0
        return self.s, {}

    def step(self, a):
        key = (self.s, int(a))
        i = int(np.searchsorted(self._cum[key], self.np_random.random()))
        i = min(i, len(self._nxt[key]) - 1)          # guard against float rounding
        self.s = int(self._nxt[key][i])
        return self.s, float(self._rew[key][i]), bool(self._done[key][i]), False, {}

    def render(self):
        out = []
        for row in range(self.nrow):
            line = ""
            for col in range(self.ncol):
                c = self.desc[row, col].decode("utf-8")
                line += "[%s]" % c if self.to_s(row, col) == self.s else " %s " % c
            out.append(line)
        return "\n".join(out)


# =========================================================================== #
#                               PLOTTING                                      #
# =========================================================================== #
def _cell_labels(env=None):
    """Letter of every square, used to mark H / G on the plots."""
    desc = env.desc if env is not None else np.asarray(
        ["SFFFF", "FFFHF", "FFFFF", "HFFFG"], dtype="c")
    return np.array([[c.decode() for c in row] for row in desc])


def _finish(fname):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def _slug(title):
    return title.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")


def plot_value_function(V, title="Value Function", env=None):
    """Heat-map of the state-value function V(s) over the 4x5 grid."""
    V = np.asarray(V, dtype=float).reshape((4, 5))
    labels = _cell_labels(env)
    annot = np.array([["%.2f\n%s" % (V[r, c], labels[r, c]) if labels[r, c] in "HG"
                       else "%.2f" % V[r, c] for c in range(5)] for r in range(4)])
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(V, annot=annot, fmt="", cmap="coolwarm", cbar=True,
                square=True, linewidths=0.5, linecolor="grey")
    plt.title(title)
    plt.xlabel("column")
    plt.ylabel("row")
    _finish(_slug(title) + ".png")


def plot_policy(P, title="Policy", env=None, V=None):
    """Arrow map of a deterministic policy; terminal squares show H / G."""
    P = np.asarray(P).reshape((4, 5))
    labels = _cell_labels(env)
    arrows = np.array([[labels[r, c] if labels[r, c] in "HG"
                        else ACTION_ARROWS[int(P[r, c])]
                        for c in range(5)] for r in range(4)])
    background = np.zeros((4, 5)) if V is None else np.asarray(V, float).reshape(4, 5)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(background, annot=arrows, fmt="", cbar=V is not None,
                cmap="coolwarm", square=True, linewidths=0.5, linecolor="grey",
                annot_kws={"size": 20})
    plt.title(title)
    plt.xlabel("column")
    plt.ylabel("row")
    _finish(_slug(title) + ".png")


def plot_q_function(Q, title="Action-Value Function", env=None):
    """One heat-map per action a of the action-value function Q(s, a)."""
    Q = np.asarray(Q, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    vmin, vmax = Q.min(), Q.max()
    for a, ax in enumerate(axes.flat):
        sns.heatmap(Q[:, a].reshape(4, 5), annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=False, square=True, linewidths=0.5, linecolor="grey",
                    vmin=vmin, vmax=vmax, ax=ax)
        ax.set_title("a = %d   %s  %s" % (a, ACTION_NAMES[a], ACTION_ARROWS[a]))
    fig.suptitle(title)
    _finish(_slug(title) + ".png")


def plot_learning_curves(curves, title="Learning curves", window=500):
    """Moving average of the undiscounted episode return."""
    plt.figure(figsize=(9, 5))
    for label, returns in curves.items():
        r = np.asarray(returns, dtype=float)
        if len(r) >= window:
            smooth = np.convolve(r, np.ones(window) / window, mode="valid")
            plt.plot(np.arange(len(smooth)) + window, smooth, label=label)
        else:
            plt.plot(r, label=label)
    plt.xlabel("episode")
    plt.ylabel("episode return (moving avg, window=%d)" % window)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    _finish(_slug(title) + ".png")


def print_policy(policy, env, title=""):
    """ASCII version of the policy, handy for the written report."""
    labels = _cell_labels(env)
    policy = np.asarray(policy).reshape(4, 5)
    print("\n" + title)
    for r in range(4):
        print("  " + "  ".join(labels[r, c] if labels[r, c] in "HG"
                               else ACTION_ARROWS[int(policy[r, c])] for c in range(5)))


# =========================================================================== #
#                 ASSIGNMENT 1 -- DYNAMIC PROGRAMMING                         #
# =========================================================================== #
def _q_from_v(P, s, V, gamma, nA):
    """One-step look-ahead: Q(s, a) for every a, given V."""
    q = np.zeros(nA)
    for a in range(nA):
        q[a] = sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in P[s][a])
    return q


def policy_evaluation(P, nS, nA, policy, gamma=GAMMA, theta=1e-10):
    """Iterative policy evaluation of a deterministic policy (in-place sweeps)."""
    V = np.zeros(nS)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            a = int(policy[s])
            V[s] = sum(p * (r + gamma * V[s2] * (not done))
                       for p, s2, r, done in P[s][a])
            delta = max(delta, abs(v_old - V[s]))
        sweeps += 1
        if delta < theta:
            return V, sweeps


def policy_iteration(env, gamma=GAMMA, theta=1e-10, verbose=True):
    """Assignment 1a -- Policy Iteration: evaluate -> improve -> repeat."""
    env = env.unwrapped
    P, nS, nA = env.P, env.nS, env.nA
    policy = np.zeros(nS, dtype=int)
    t0 = time.perf_counter()
    n_iter, total_sweeps = 0, 0

    while True:
        V, sweeps = policy_evaluation(P, nS, nA, policy, gamma, theta)
        total_sweeps += sweeps
        n_iter += 1
        stable = True
        for s in range(nS):
            old_a = policy[s]
            policy[s] = int(np.argmax(_q_from_v(P, s, V, gamma, nA)))
            if old_a != policy[s]:
                stable = False
        if stable:                      # greedy policy no longer changes -> optimal
            break

    elapsed = time.perf_counter() - t0
    RUN_STATS["Policy Iteration"] = {"iterations": n_iter, "sweeps": total_sweeps,
                                     "time_s": elapsed}
    if verbose:
        print("[Policy Iteration] converged after %d policy improvements "
              "(%d evaluation sweeps) -- %.1f ms" % (n_iter, total_sweeps, elapsed * 1e3))
    return policy, V


def value_iteration(env, gamma=GAMMA, theta=1e-10, verbose=True):
    """Assignment 1b -- Value Iteration: repeated Bellman optimality backup."""
    env = env.unwrapped
    P, nS, nA = env.P, env.nS, env.nA
    V = np.zeros(nS)
    t0 = time.perf_counter()
    sweeps = 0

    while True:
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            V[s] = np.max(_q_from_v(P, s, V, gamma, nA))
            delta = max(delta, abs(v_old - V[s]))
        sweeps += 1
        if delta < theta:
            break

    # A single greedy pass extracts the optimal policy from V*.
    policy = np.array([int(np.argmax(_q_from_v(P, s, V, gamma, nA)))
                       for s in range(nS)])

    elapsed = time.perf_counter() - t0
    RUN_STATS["Value Iteration"] = {"iterations": sweeps, "sweeps": sweeps,
                                    "time_s": elapsed}
    if verbose:
        print("[Value Iteration]  converged after %d sweeps -- %.1f ms"
              % (sweeps, elapsed * 1e3))
    return policy, V


# =========================================================================== #
#                    MODEL-FREE CONTROL (Assignments 2-4)                     #
# =========================================================================== #
def epsilon_greedy(Q, state, epsilon):
    """Behaviour policy: explore w.p. epsilon, otherwise act greedily w.r.t. Q."""
    if RNG.random() < epsilon:
        return int(RNG.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


def glie_epsilon(k, eps_start=1.0, eps_decay=1e-3):
    """GLIE schedule: eps_k = eps_start / (1 + decay*k).

    It decays to 0 (so the policy becomes greedy in the limit) but slowly
    enough that every state-action pair keeps being visited infinitely often.
    """
    return eps_start / (1.0 + eps_decay * k)


def greedy_policy(Q):
    return np.argmax(Q, axis=1)


def monte_carlo_glie(env, alpha=0.02, gamma=GAMMA, num_episodes=200000,
                     eps_start=1.0, eps_decay=2e-4, max_steps=200, verbose=True):
    """Assignment 2 -- first-visit GLIE Monte-Carlo control, constant alpha.

    Q(s,a) <- Q(s,a) + alpha * (G_t - Q(s,a))   at the FIRST visit of (s,a).
    """
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    returns_hist = np.zeros(num_episodes)
    t0 = time.perf_counter()

    for k in range(num_episodes):
        eps = glie_epsilon(k, eps_start, eps_decay)

        # ---- 1. generate one episode with the current eps-greedy policy ----
        s, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            s2, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s2
            if terminated or truncated:
                break
        returns_hist[k] = sum(step[2] for step in episode)

        # ---- 2. first-visit indices of every (s,a) pair --------------------
        first = {}
        for t, (s_t, a_t, _) in enumerate(episode):
            first.setdefault((s_t, a_t), t)

        # ---- 3. backward pass: G_t = r_t + gamma * G_{t+1} -----------------
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = r_t + gamma * G
            if first[(s_t, a_t)] == t:
                Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])

    elapsed = time.perf_counter() - t0
    eps_end = glie_epsilon(num_episodes, eps_start, eps_decay)
    RUN_STATS["Monte Carlo (GLIE)"] = {"episodes": num_episodes, "time_s": elapsed,
                                       "final_eps": eps_end}
    if verbose:
        print("[MC GLIE]    %d episodes, final eps = %.4f -- %.1f s"
              % (num_episodes, eps_end, elapsed))
    return greedy_policy(Q), Q, returns_hist


def sarsa(env, alpha=0.02, gamma=GAMMA, num_episodes=200000,
          eps_start=1.0, eps_decay=2e-4, max_steps=200, verbose=True):
    """Assignment 3 -- Sarsa, on-policy TD(0) control.

    Q(s,a) <- Q(s,a) + alpha * (r + gamma*Q(s',a') - Q(s,a)),
    where a' comes from the same eps-greedy policy that is being improved.
    """
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    returns_hist = np.zeros(num_episodes)
    t0 = time.perf_counter()

    for k in range(num_episodes):
        eps = glie_epsilon(k, eps_start, eps_decay)
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, eps)
        total = 0.0
        for _ in range(max_steps):
            s2, r, terminated, truncated, _ = env.step(a)
            total += r
            if terminated:
                # No bootstrapping past a terminal state: V(terminal) = 0.
                Q[s, a] += alpha * (r - Q[s, a])
                break
            a2 = epsilon_greedy(Q, s2, eps)
            Q[s, a] += alpha * (r + gamma * Q[s2, a2] - Q[s, a])
            s, a = s2, a2
            if truncated:
                break
        returns_hist[k] = total

    elapsed = time.perf_counter() - t0
    eps_end = glie_epsilon(num_episodes, eps_start, eps_decay)
    RUN_STATS["Sarsa"] = {"episodes": num_episodes, "time_s": elapsed,
                          "final_eps": eps_end}
    if verbose:
        print("[Sarsa]      %d episodes, final eps = %.4f -- %.1f s"
              % (num_episodes, eps_end, elapsed))
    return greedy_policy(Q), Q, returns_hist


def q_learning(env, alpha=0.02, gamma=GAMMA, num_episodes=200000,
               eps_start=1.0, eps_decay=2e-4, max_steps=200, verbose=True):
    """Assignment 4 -- Q-Learning, off-policy TD(0) control.

    Q(s,a) <- Q(s,a) + alpha * (r + gamma*max_a' Q(s',a') - Q(s,a)).
    The target is greedy while the behaviour policy stays eps-greedy.
    """
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    returns_hist = np.zeros(num_episodes)
    t0 = time.perf_counter()

    for k in range(num_episodes):
        eps = glie_epsilon(k, eps_start, eps_decay)
        s, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, eps)
            s2, r, terminated, truncated, _ = env.step(a)
            total += r
            target = r if terminated else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            if terminated or truncated:
                break
        returns_hist[k] = total

    elapsed = time.perf_counter() - t0
    eps_end = glie_epsilon(num_episodes, eps_start, eps_decay)
    RUN_STATS["Q-Learning"] = {"episodes": num_episodes, "time_s": elapsed,
                               "final_eps": eps_end}
    if verbose:
        print("[Q-Learning] %d episodes, final eps = %.4f -- %.1f s"
              % (num_episodes, eps_end, elapsed))
    return greedy_policy(Q), Q, returns_hist


# =========================================================================== #
#                              EVALUATION                                     #
# =========================================================================== #
def policy_agreement(policy, reference, env):
    """Share of non-terminal states where the learned policy matches DP."""
    env = env.unwrapped
    mask = ~env.terminal_states
    return float(np.mean(np.asarray(policy)[mask] == np.asarray(reference)[mask]))


def exact_policy_value(env, policy, gamma=GAMMA):
    """Exact V^pi via policy evaluation, averaged over the random start states.

    This is noise-free (unlike a Monte-Carlo roll-out estimate), so two runs
    that produce the same policy always get exactly the same score.
    """
    env = env.unwrapped
    V_pi, _ = policy_evaluation(env.P, env.nS, env.nA, policy, gamma)
    return V_pi, float(np.mean(V_pi[env.start_states]))


def report_policy_differences(policy, reference, V_star, env, name=""):
    """List the states where a learned policy differs from the DP optimum,
    together with the optimality gap Q*(s,pi*) - Q*(s,pi) of that choice.

    A gap close to 0 means the two actions are (almost) equally good, i.e. the
    disagreement costs nothing -- which is what happens for most differences.
    """
    env = env.unwrapped
    rows = []
    for s in range(env.nS):
        if env.terminal_states[s] or policy[s] == reference[s]:
            continue
        q = _q_from_v(env.P, s, V_star, GAMMA, env.nA)
        rows.append((s, s // env.ncol, s % env.ncol, ACTION_ARROWS[int(reference[s])],
                     ACTION_ARROWS[int(policy[s])], q[int(reference[s])] - q[int(policy[s])]))
    if not rows:
        print("  %s matches the DP optimal policy in every state." % name)
        return
    print("  %s differs from pi* in %d of %d non-terminal states:"
          % (name, len(rows), int(np.sum(~env.terminal_states))))
    print("     state (row,col)   pi*   learned   Q* gap")
    for s, r, c, a_ref, a_pol, gap in rows:
        print("     %2d    (%d,%d)      %s      %s     %.4f" % (s, r, c, a_ref, a_pol, gap))


def evaluate_policy(env, policy, gamma=GAMMA, episodes=3000, max_steps=200):
    """Empirical discounted return and success rate of a greedy policy."""
    env_u = env.unwrapped
    returns, successes = np.zeros(episodes), 0
    for i in range(episodes):
        s, _ = env.reset()
        G, disc = 0.0, 1.0
        for _ in range(max_steps):
            s, r, terminated, truncated, _ = env.step(int(policy[s]))
            G += disc * r
            disc *= gamma
            if terminated:
                successes += int(env_u.goal_states[s])
                break
            if truncated:
                break
        returns[i] = G
    return returns.mean(), successes / episodes


# =========================================================================== #
#                                  MAIN                                       #
# =========================================================================== #
if __name__ == "__main__":

    # Register the custom Frozen Lake environment
    gym.envs.registration.register(
        id="CustomFrozenLake-v0",
        entry_point=__name__ + ":CustomFrozenLakeEnv",
        max_episode_steps=200,          # safety net -> truncation, not termination
    )

    # Create the environment
    env = gym.make("CustomFrozenLake-v0", is_slippery=True, slip_prob=SLIP_PROB,
                   disable_env_checker=True)
    env.reset(seed=0)
    set_seed(0)

    # Shared hyper-parameters of the three model-free methods, so that
    # Assignments 2-4 are compared under identical conditions.
    ALPHA = 0.02          # constant step size (required by Assignment 2)
    N_EPISODES = 200000
    EPS_DECAY = 2e-4      # eps_k = 1 / (1 + EPS_DECAY*k)  ->  eps_final ~ 0.024

    print("Grid  (S=start, F=frozen, H=hole, G=goal):")
    print(env.unwrapped.render())
    print("\ngamma=%.2f   step reward=%.2f   P(intended)=%.1f   P(slip to each side)=%.1f\n"
          % (GAMMA, STEP_REWARD, 1 - SLIP_PROB, SLIP_PROB / 2))

    # ---------------- Assignment 1: Dynamic Programming ------------------- #
    print("=" * 66)
    print("ASSIGNMENT 1 -- Dynamic Programming")
    print("=" * 66)
    policy_pi, V_pi = policy_iteration(env, gamma=GAMMA)
    policy_vi, V_vi = value_iteration(env, gamma=GAMMA)

    print("max |V_PI - V_VI| = %.2e   (identical optimal policies: %s)"
          % (np.max(np.abs(V_pi - V_vi)), np.array_equal(policy_pi, policy_vi)))
    print_policy(policy_vi, env.unwrapped, "Optimal policy pi* (DP):")

    plot_value_function(V_vi, "Value Function (Value Iteration)", env.unwrapped)
    plot_policy(policy_vi, "Policy (Value Iteration)", env.unwrapped, V=V_vi)
    plot_value_function(V_pi, "Value Function (Policy Iteration)", env.unwrapped)
    plot_policy(policy_pi, "Policy (Policy Iteration)", env.unwrapped, V=V_pi)

    # ---------------- Assignment 2: Monte Carlo GLIE ---------------------- #
    print("\n" + "=" * 66)
    print("ASSIGNMENT 2 -- Monte Carlo (GLIE, first-visit)")
    print("=" * 66)
    policy_mc, Q_mc, ret_mc = monte_carlo_glie(env, alpha=ALPHA, gamma=GAMMA,
                                               num_episodes=N_EPISODES,
                                               eps_decay=EPS_DECAY)
    print_policy(policy_mc, env.unwrapped, "Policy (Monte Carlo GLIE):")
    plot_value_function(np.max(Q_mc, axis=1),
                        "Value Function (Monte Carlo GLIE)", env.unwrapped)
    plot_q_function(Q_mc, "Action-Value Function (Monte Carlo GLIE)", env.unwrapped)
    plot_policy(policy_mc, "Policy (Monte Carlo GLIE)", env.unwrapped,
                V=np.max(Q_mc, axis=1))

    # ---------------- Assignment 3: Sarsa --------------------------------- #
    print("\n" + "=" * 66)
    print("ASSIGNMENT 3 -- Sarsa")
    print("=" * 66)
    policy_sarsa, Q_sarsa, ret_sarsa = sarsa(env, alpha=ALPHA, gamma=GAMMA,
                                             num_episodes=N_EPISODES,
                                             eps_decay=EPS_DECAY)
    print_policy(policy_sarsa, env.unwrapped, "Policy (Sarsa):")
    plot_value_function(np.max(Q_sarsa, axis=1), "Value Function (Sarsa)", env.unwrapped)
    plot_q_function(Q_sarsa, "Action-Value Function (Sarsa)", env.unwrapped)
    plot_policy(policy_sarsa, "Policy (Sarsa)", env.unwrapped,
                V=np.max(Q_sarsa, axis=1))

    # ---------------- Assignment 4: Q-Learning ---------------------------- #
    print("\n" + "=" * 66)
    print("ASSIGNMENT 4 -- Q-Learning")
    print("=" * 66)
    policy_ql, Q_ql, ret_ql = q_learning(env, alpha=ALPHA, gamma=GAMMA,
                                         num_episodes=N_EPISODES,
                                         eps_decay=EPS_DECAY)
    print_policy(policy_ql, env.unwrapped, "Policy (Q-Learning):")
    plot_value_function(np.max(Q_ql, axis=1), "Value Function (Q-Learning)",
                        env.unwrapped)
    plot_q_function(Q_ql, "Action-Value Function (Q-Learning)", env.unwrapped)
    plot_policy(policy_ql, "Policy (Q-Learning)", env.unwrapped,
                V=np.max(Q_ql, axis=1))

    plot_learning_curves({"Monte Carlo (GLIE)": ret_mc, "Sarsa": ret_sarsa,
                          "Q-Learning": ret_ql},
                         "Learning curves (episode return)")

    # ---------------- Final comparison ------------------------------------ #
    print("\n" + "=" * 66)
    print("COMPARISON   (reference = the DP optimum)")
    print("=" * 66)
    # V*(s) averaged over the start states = the best achievable score.
    _, v_star_mean = exact_policy_value(env, policy_vi, GAMMA)

    header = ("%-22s%12s%14s%12s%12s%9s%9s"
              % ("method", "max|V-V*|", "policy match", "E[V^pi]", "value loss",
                 "success", "time [s]"))
    print(header)
    print("-" * len(header))
    runs = [("Policy Iteration", policy_pi, V_pi),
            ("Value Iteration", policy_vi, V_vi),
            ("Monte Carlo (GLIE)", policy_mc, np.max(Q_mc, axis=1)),
            ("Sarsa", policy_sarsa, np.max(Q_sarsa, axis=1)),
            ("Q-Learning", policy_ql, np.max(Q_ql, axis=1))]
    for name, pol, V in runs:
        err = np.max(np.abs(np.asarray(V) - V_vi))          # quality of the value estimate
        match = policy_agreement(pol, policy_vi, env)        # exact argmax agreement
        _, v_pi = exact_policy_value(env, pol, GAMMA)        # exact quality of the policy
        _, success = evaluate_policy(env, pol, gamma=GAMMA, episodes=5000)
        t = RUN_STATS.get(name, {}).get("time_s", float("nan"))
        print("%-22s%12.4f%13.0f%%%12.4f%12.4f%8.1f%%%9.2f"
              % (name, err, 100 * match, v_pi, v_star_mean - v_pi, 100 * success, t))

    print("""
  max|V-V*|    : largest error of the estimated value function vs. the DP optimum
  policy match : share of non-terminal states whose greedy action equals pi*
  E[V^pi]      : exact expected discounted return of the learned policy,
                 averaged over the random start states (computed analytically)
  value loss   : E[V*] - E[V^pi]; 0 means the policy is effectively optimal""")

    # Where -- and how much -- do the model-free policies actually disagree?
    print("\n" + "=" * 66)
    print("WHERE THE MODEL-FREE POLICIES DIFFER FROM pi*")
    print("=" * 66)
    for name, pol, _ in runs[2:]:
        report_policy_differences(pol, policy_vi, V_vi, env, name)
        print()

    print("All figures were written to: %s" % PLOT_DIR)
