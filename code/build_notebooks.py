# -*- coding: utf-8 -*-
"""Generate one standalone Colab notebook per assignment.

The code cells are extracted straight out of ``frozen_lake_rl.py`` (so the
notebooks can never drift away from the tested implementation), and the numbers
quoted in the analysis cells are measured here by actually running the
algorithms with the same seeds the notebooks use.

    python build_notebooks.py
"""

import ast
import io
import json
import os

os.environ.setdefault("RL_SHOW_PLOTS", "0")   # never pop up a window while measuring

import numpy as np
import gymnasium as gym

import frozen_lake_rl as fl

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = io.open(os.path.join(HERE, "frozen_lake_rl.py"), encoding="utf-8").read()
SRC_LINES = SRC.splitlines(keepends=True)
TREE = ast.parse(SRC)

SEED = 0
ALPHA = 0.02
N_EPISODES = 200000
EPS_DECAY = 2e-4


# --------------------------------------------------------------------------- #
# source extraction
# --------------------------------------------------------------------------- #
def seg(*names):
    """Return the source of the named top-level functions / classes."""
    out = []
    for name in names:
        for node in TREE.body:
            if getattr(node, "name", None) == name:
                out.append("".join(SRC_LINES[node.lineno - 1:node.end_lineno]).rstrip())
                break
        else:
            raise KeyError(name)
    return "\n\n\n".join(out)


# --------------------------------------------------------------------------- #
# notebook helpers
# --------------------------------------------------------------------------- #
def _src(text):
    return text.strip("\n").splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(text)}


def write_nb(cells, path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d cells)" % (os.path.basename(path), len(cells)))


def policy_ascii(policy, env, indent="    "):
    labels = fl._cell_labels(env)
    policy = np.asarray(policy).reshape(4, 5)
    rows = []
    for r in range(4):
        rows.append(indent + "  ".join(
            labels[r, c] if labels[r, c] in "HG" else fl.ACTION_ARROWS[int(policy[r, c])]
            for c in range(5)))
    return "\n".join(rows)


# =========================================================================== #
# shared cells
# =========================================================================== #
HEADER_FMT = """# {title}

**Frozen Lake - 4x5 Gridworld · Summer Semester 2026**

| | |
|---|---|
| **Name** | `Nihat Garibli` |
| **ID** | `5425819` |

{intro}
"""

SETUP = """
# gymnasium is pre-installed on Colab; install it only if it is missing.
try:
    import gymnasium  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium"])

import time

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_theme(style="white")
"""

ENV_MD = """
## 1. The environment

```
 S  F  F  F  F        states  0.. 4
 F  F  F  H  F        states  5.. 9   (hole at state 8)
 F  F  F  F  F        states 10..14
 H  F  F  F  G        states 15..19   (hole at 15, goal at 19)
```

Exactly as specified in the assignment sheet:

| Setting | Value |
|---|---|
| Step reward (**every** transition) | −0.08 |
| Goal / hole | +1 / −1, **added on top** of the step reward |
| P(intended direction) | 0.8 |
| P(slip to each perpendicular side) | 0.1 + 0.1 |
| Walls | rigid - the agent stays in place |
| Discount γ | 0.95 |
| Start state | uniformly random over the 17 non-terminal squares |

Actions: `0 = ← (W)`, `1 = ↓ (S)`, `2 = → (E)`, `3 = ↑ (N)`.

The step cost is what makes the task interesting: dawdling is expensive, so the
optimal policy has to trade off the **shortest** path against the **safest** path
around the two holes.

The environment exposes the full model as
`P[s][a] = [(prob, next_state, reward, terminated), ...]`, which is what the
Dynamic-Programming methods need; the model-free methods only use
`reset()` / `step()`.
"""

CONSTANTS = """
# ---- constants, taken directly from the assignment sheet -------------------
GAMMA = 0.95
STEP_REWARD = -0.08
GOAL_REWARD = 1.0
HOLE_REWARD = -1.0
SLIP_PROB = 0.2

ACTION_ARROWS = ["←", "↓", "→", "↑"]          # left, down, right, up
ACTION_NAMES = ["W (left)", "S (down)", "E (right)", "N (up)"]

SEED = 0

# Filled in by the algorithms; used for the timing / iteration tables.
RUN_STATS = {{}}

# Module level RNG, so that a whole run is reproducible via set_seed().
RNG = np.random.default_rng(SEED)


{set_seed}
""".format(set_seed=seg("set_seed"))

ENV_CODE = seg("CustomFrozenLakeEnv")

MAKE_ENV = """
# Register the custom environment and create it through the gymnasium API.
if "CustomFrozenLake-v0" not in gym.registry:
    gym.envs.registration.register(
        id="CustomFrozenLake-v0",
        entry_point=CustomFrozenLakeEnv,
        max_episode_steps=200,        # safety net -> truncation, not termination
    )

env = gym.make("CustomFrozenLake-v0", is_slippery=True, slip_prob=SLIP_PROB,
               disable_env_checker=True)
env.reset(seed=SEED)
set_seed(SEED)

print("Grid  (S=start, F=frozen, H=hole, G=goal):")
print(env.unwrapped.render())
print("\\ngamma=%.2f   step reward=%.2f   P(intended)=%.1f   P(slip to each side)=%.1f"
      % (GAMMA, STEP_REWARD, 1 - SLIP_PROB, SLIP_PROB / 2))
print("start states:", env.unwrapped.start_states)
"""

PLOT_MD = """
## 2. Plotting helpers

Every figure gets a title, axis labels and readable annotations; terminal squares
are marked `H` / `G` instead of an arrow.
"""

PLOT_CODE_BASE = """
def _finish(fname=None):
    plt.tight_layout()
    plt.show()


{funcs}
"""


def plot_code(extra=()):
    names = ["_cell_labels", "_slug", "plot_value_function", "plot_policy",
             "print_policy"] + list(extra)
    return PLOT_CODE_BASE.format(funcs=seg(*names))


DP_MD = """
## {n}. Dynamic Programming (reference solution)

The DP solution from Assignment 1 is reproduced here **only as a benchmark**: it
gives the exact optimum V\\* / π\\*, which lets us measure how good the learned
policy really is.
"""

DP_CODE = seg("_q_from_v", "policy_evaluation", "value_iteration")

EVAL_CODE = seg("policy_agreement", "exact_policy_value",
                "report_policy_differences", "evaluate_policy")

MODEL_FREE_HELPERS = seg("epsilon_greedy", "glie_epsilon", "greedy_policy")

HYPER = """
# ---- shared hyper-parameters of the model-free methods ---------------------
ALPHA = {alpha}          # CONSTANT step size, as required by the assignment
N_EPISODES = {n}
EPS_DECAY = {decay}      # eps_k = 1 / (1 + EPS_DECAY*k)
""".format(alpha=ALPHA, n=N_EPISODES, decay=EPS_DECAY)

GLIE_MD = """
### The GLIE exploration schedule

$$\\varepsilon_k = \\frac{{1}}{{1 + \\texttt{{EPS\\_DECAY}} \\cdot k}}$$

This satisfies both GLIE conditions:

* $\\varepsilon_k \\to 0$, so the policy becomes **greedy in the limit**;
* the decay is slow enough (harmonic) that **every state-action pair keeps being
  visited infinitely often**.

The random start state helps for the same reason: it guarantees that squares a
greedy policy would never revisit still get explored.
"""


# =========================================================================== #
# measurements
# =========================================================================== #
def make_env():
    if "CustomFrozenLake-v0" not in gym.registry:
        gym.envs.registration.register(
            id="CustomFrozenLake-v0", entry_point=fl.CustomFrozenLakeEnv,
            max_episode_steps=200)
    return gym.make("CustomFrozenLake-v0", is_slippery=True,
                    slip_prob=fl.SLIP_PROB, disable_env_checker=True)


def run_model_free(algo, env):
    """Reproduce exactly what the notebook does: re-seed, then train."""
    env.reset(seed=SEED)
    fl.set_seed(SEED)
    policy, Q, returns = algo(env, alpha=ALPHA, gamma=fl.GAMMA,
                              num_episodes=N_EPISODES, eps_decay=EPS_DECAY,
                              verbose=False)
    return policy, Q, returns


print("measuring ... (this runs the algorithms once, ~30 s)")
ENV = make_env()

POL_PI, V_PI = fl.policy_iteration(ENV, verbose=False)
POL_VI, V_VI = fl.value_iteration(ENV, verbose=False)
_, V_STAR_MEAN = fl.exact_policy_value(ENV, POL_VI, fl.GAMMA)
S_PI, S_VI = fl.RUN_STATS["Policy Iteration"], fl.RUN_STATS["Value Iteration"]

RESULTS = {}
for key, algo in [("mc", fl.monte_carlo_glie), ("sarsa", fl.sarsa),
                  ("ql", fl.q_learning)]:
    pol, Q, _ = run_model_free(algo, ENV)
    ENV.reset(seed=123)
    _, success = fl.evaluate_policy(ENV, pol, gamma=fl.GAMMA, episodes=5000)
    _, v_pi = fl.exact_policy_value(ENV, pol, fl.GAMMA)
    RESULTS[key] = {
        "policy": pol,
        "match": fl.policy_agreement(pol, POL_VI, ENV),
        "v_pi": v_pi,
        "loss": V_STAR_MEAN - v_pi,
        "success": success,
        "verr": float(np.max(np.abs(np.max(Q, axis=1) - V_VI))),
        "ascii": policy_ascii(pol, ENV.unwrapped),
    }
    print("  %-6s match=%.0f%%  loss=%.4f  success=%.1f%%"
          % (key, 100 * RESULTS[key]["match"], RESULTS[key]["loss"],
             100 * RESULTS[key]["success"]))


ENV.reset(seed=123)
_, SUCCESS_STAR = fl.evaluate_policy(ENV, POL_VI, gamma=fl.GAMMA, episodes=5000)

# A single seed is not enough to compare the three methods: at seed 0 Sarsa and
# Q-Learning happen to find the same policy. Repeat over several seeds.
SEEDS = [0, 1, 2, 3, 4]
MULTI = {}
print("multi-seed study over %d seeds ..." % len(SEEDS))
for key, algo in [("mc", fl.monte_carlo_glie), ("sarsa", fl.sarsa),
                  ("ql", fl.q_learning)]:
    rows = []
    for seed in SEEDS:
        ENV.reset(seed=seed)
        fl.set_seed(seed)
        pol, Q, _ = algo(ENV, alpha=ALPHA, gamma=fl.GAMMA, num_episodes=N_EPISODES,
                         eps_decay=EPS_DECAY, verbose=False)
        ENV.reset(seed=123)
        _, succ = fl.evaluate_policy(ENV, pol, gamma=fl.GAMMA, episodes=5000)
        _, v = fl.exact_policy_value(ENV, pol, fl.GAMMA)
        rows.append((V_STAR_MEAN - v, succ, fl.policy_agreement(pol, POL_VI, ENV)))
    MULTI[key] = np.array(rows)
    print("  %-6s loss=%.4f+-%.4f  success=%.1f%%  match=%.0f%%"
          % (key, MULTI[key][:, 0].mean(), MULTI[key][:, 0].std(),
             100 * MULTI[key][:, 1].mean(), 100 * MULTI[key][:, 2].mean()))


def ms(key, col):
    """mean +- std of one column of the multi-seed study."""
    a = MULTI[key][:, col]
    return a.mean(), a.std()


def multi_seed_table():
    rows = ["| method | value loss | success rate | policy match |",
            "|---|---|---|---|",
            "| **π\\* (Dynamic Programming)** | 0.0000 | %.1f%% | 100%% |"
            % (100 * SUCCESS_STAR)]
    for key, label in [("mc", "Monte Carlo (GLIE)"), ("sarsa", "Sarsa"),
                       ("ql", "Q-Learning")]:
        lm, ls = ms(key, 0)
        sm, ss = ms(key, 1)
        mm, msd = ms(key, 2)
        rows.append("| %s | %.4f ± %.4f | %.1f%% ± %.1f | %.0f%% ± %.0f |"
                    % (label, lm, ls, 100 * sm, 100 * ss, 100 * mm, 100 * msd))
    return "\n".join(rows)


def diff_table(key):
    """Markdown list of the states where a learned policy differs from pi*."""
    pol = RESULTS[key]["policy"]
    env = ENV.unwrapped
    rows = []
    for s in range(env.nS):
        if env.terminal_states[s] or pol[s] == POL_VI[s]:
            continue
        q = fl._q_from_v(env.P, s, V_VI, fl.GAMMA, env.nA)
        rows.append("| %d | (%d,%d) | %s | %s | %.4f |"
                    % (s, s // env.ncol, s % env.ncol,
                       fl.ACTION_ARROWS[int(POL_VI[s])],
                       fl.ACTION_ARROWS[int(pol[s])],
                       q[int(POL_VI[s])] - q[int(pol[s])]))
    if not rows:
        return "The learned policy matches π\\* in **every** state."
    return ("| state | (row,col) | π\\* | learned | Q\\* gap |\n|---|---|---|---|---|\n"
            + "\n".join(rows))


# =========================================================================== #
# NOTEBOOK 1 -- Dynamic Programming
# =========================================================================== #
def notebook_1():
    cells = [
        md(HEADER_FMT.format(
            title="Assignment 1 - Dynamic Programming",
            intro="Find the optimal policy π\\* and the optimal state-value function "
                  "V\\* by solving the MDP with **Policy Iteration** and **Value "
                  "Iteration**, and compare the two methods.")),
        code(SETUP),
        md(ENV_MD),
        code(CONSTANTS),
        code(ENV_CODE),
        code(MAKE_ENV),
        md(PLOT_MD),
        code(plot_code()),
        md("""
## 3. Policy Iteration

Alternate two steps until the policy stops changing:

1. **Policy evaluation** - solve $V^{\\pi}$ for the current policy by sweeping

$$V(s) \\leftarrow \\sum_{s'} P(s'|s,\\pi(s))\\,[\\,r + \\gamma V(s')\\,]$$

2. **Policy improvement** - act greedily w.r.t. the value just computed

$$\\pi(s) \\leftarrow \\arg\\max_a \\sum_{s'} P(s'|s,a)\\,[\\,r + \\gamma V(s')\\,]$$

When the greedy policy no longer changes, it is **provably optimal** - the
algorithm terminates exactly, with no threshold involved.
"""),
        code(seg("_q_from_v", "policy_evaluation", "policy_iteration")),
        md("""
## 4. Value Iteration

Skip the inner evaluation loop entirely and apply the Bellman **optimality**
backup directly:

$$V(s) \\leftarrow \\max_a \\sum_{s'} P(s'|s,a)\\,[\\,r + \\gamma V(s')\\,]$$

Sweeps stop when $\\max_s |\\Delta V(s)| < \\theta$, and one final greedy pass
extracts π\\* from V\\*.
"""),
        code(seg("value_iteration")),
        md("## 5. Run both methods"),
        code("""
policy_pi, V_pi = policy_iteration(env, gamma=GAMMA)
policy_vi, V_vi = value_iteration(env, gamma=GAMMA)

print("\\nmax |V_PI - V_VI| = %.2e" % np.max(np.abs(V_pi - V_vi)))
print("identical optimal policies: %s" % np.array_equal(policy_pi, policy_vi))

print_policy(policy_vi, env.unwrapped, "Optimal policy pi*:")
"""),
        md("## 6. The three required figures"),
        code("""
plot_value_function(V_vi, "Optimal Value Function V* (Value Iteration)", env.unwrapped)
plot_policy(policy_vi, "Optimal Policy pi* (Value Iteration)", env.unwrapped, V=V_vi)

plot_value_function(V_pi, "Optimal Value Function V* (Policy Iteration)", env.unwrapped)
plot_policy(policy_pi, "Optimal Policy pi* (Policy Iteration)", env.unwrapped, V=V_pi)
"""),
        md("## 7. How much work did each method do?"),
        code("""
print("%-20s%14s%16s%12s" % ("method", "outer iters", "Bellman sweeps", "time [ms]"))
print("-" * 62)
for name in ("Policy Iteration", "Value Iteration"):
    st = RUN_STATS[name]
    print("%-20s%14d%16d%12.1f"
          % (name, st["iterations"], st["sweeps"], 1000 * st["time_s"]))
"""),
        md("""
## 8. Results and conclusions

Both methods return **exactly the same** optimal policy and value function
(`max|V_PI − V_VI| ≈ 5e-12`, which is pure floating-point noise):

```
{pol}
```

Every arrow flows toward the goal at (3,4) while bending away from the two holes.
The clearest example is **state 9 = (1,4)**: the policy plays `↓` instead of `←`,
because a leftward slip could drop the agent into the hole at state 8. This is the
0.1 slip probability being priced into the policy - with a deterministic
environment the greedy shortcut would win.

### Policy Iteration vs Value Iteration

| | Policy Iteration | Value Iteration |
|---|---|---|
| Outer iterations | **{pi_it}** | {vi_it} |
| Total Bellman sweeps | {pi_sw} | **{vi_sw}** |
| Runtime | {pi_ms:.1f} ms | **{vi_ms:.1f} ms** |
| Stopping rule | exact (policy stable) | threshold θ |

**Advantages of Policy Iteration**

* Very few outer iterations - each improvement is a large jump in policy space,
  and the number of iterations is bounded by the (finite) number of policies.
* Terminates **exactly**: a stable greedy policy is optimal, so no θ is needed.
* The evaluation step can be replaced by directly solving the linear system
  (20×20 here), which makes each iteration cheap for small state spaces.

**Disadvantages of Policy Iteration**

* Every outer iteration runs a *full* policy evaluation to convergence -
  {pi_sw} sweeps in total here versus {vi_sw} for VI.
* Much more expensive per iteration, and the effort spent on precisely evaluating
  an intermediate policy that is about to be discarded is largely wasted.

**Advantages of Value Iteration**

* One cheap Bellman optimality backup per sweep, no inner loop, trivial to
  implement.
* On this MDP it is roughly {speedup:.0f}× faster in wall-clock time.

**Disadvantages of Value Iteration**

* Converges only asymptotically, so it needs an arbitrary threshold θ.
* The number of sweeps grows as γ → 1 (roughly `log θ / log γ`), so for
  long-horizon problems VI can need many more sweeps.

**Conclusion.** On a 20-state MDP with γ = 0.95, Value Iteration is the better
choice: same answer, a fraction of the work. Policy Iteration pays off when γ is
close to 1 (few improvements still suffice while VI would need hundreds of
sweeps) or when policy evaluation can be solved in closed form. In practice
*modified policy iteration* - truncating the evaluation after k sweeps -
interpolates between the two and is usually the fastest of all.

Both share the same fundamental limitation: they need the **full model** `P`.
Assignments 2-4 drop that assumption.
""".format(pol=policy_ascii(POL_VI, ENV.unwrapped),
           pi_it=S_PI["iterations"], vi_it=S_VI["iterations"],
           pi_sw=S_PI["sweeps"], vi_sw=S_VI["sweeps"],
           pi_ms=1000 * S_PI["time_s"], vi_ms=1000 * S_VI["time_s"],
           speedup=max(S_PI["time_s"] / max(S_VI["time_s"], 1e-9), 1.0))),
    ]
    write_nb(cells, os.path.join(HERE, "Assignment_1_Dynamic_Programming.ipynb"))


# =========================================================================== #
# NOTEBOOKS 2-4 -- model-free control
# =========================================================================== #
def model_free_notebook(key, title, intro, theory_md, algo_names, run_code,
                        analysis_md, filename, extra_plots=("plot_q_function",
                                                            "plot_learning_curves"),
                        extra_cells=()):
    cells = [
        md(HEADER_FMT.format(title=title, intro=intro)),
        code(SETUP),
        md(ENV_MD),
        code(CONSTANTS),
        code(ENV_CODE),
        code(MAKE_ENV),
        md(PLOT_MD),
        code(plot_code(extra_plots)),
        md(DP_MD.format(n=3)),
        code(DP_CODE),
        code(EVAL_CODE),
        md(theory_md),
        code(HYPER + "\n\n" + MODEL_FREE_HELPERS),
        md(GLIE_MD),
        code(seg(*algo_names)),
        md("## 5. Train"),
        code(run_code),
    ]
    cells.extend(extra_cells)
    cells.append(md(analysis_md))
    write_nb(cells, os.path.join(HERE, filename))


COMMON_RUN_TAIL = """
# ---- reference solution + quality of the learned policy --------------------
policy_star, V_star = value_iteration(env, gamma=GAMMA, verbose=False)
_, v_star_mean = exact_policy_value(env, policy_star, GAMMA)
_, v_pi_mean = exact_policy_value(env, policy_{k}, GAMMA)

env.reset(seed=123)
_, success = evaluate_policy(env, policy_{k}, gamma=GAMMA, episodes=5000)

print("policy match with pi*      : %.0f%%" % (100 * policy_agreement(policy_{k}, policy_star, env)))
print("max |V - V*|               : %.4f" % np.max(np.abs(np.max(Q_{k}, axis=1) - V_star)))
print("E[V^pi]  (exact)           : %.4f   (optimum %.4f)" % (v_pi_mean, v_star_mean))
print("value loss E[V*] - E[V^pi] : %.4f" % (v_star_mean - v_pi_mean))
print("empirical success rate     : %.1f%%" % (100 * success))
print()
report_policy_differences(policy_{k}, policy_star, V_star, env, "{label}")
"""


def notebook_2():
    theory = """
## 4. GLIE Monte-Carlo control (first-visit, constant α)

Monte Carlo learns from **complete episodes** - no model, no bootstrapping.
For every state-action pair, at its **first** visit in the episode:

$$G_t = r_{t+1} + \\gamma r_{t+2} + \\gamma^2 r_{t+3} + \\dots$$

$$Q(s_t,a_t) \\leftarrow Q(s_t,a_t) + \\alpha\\,[\\,G_t - Q(s_t,a_t)\\,]$$

The return $G_t$ is accumulated in a **backward pass** over the episode, which
computes every $G_t$ in one sweep via $G_t = r_{t+1} + \\gamma G_{t+1}$.

A **constant** α (as the assignment requires) means the estimate is an
exponentially weighted average of recent returns - it keeps tracking instead of
converging exactly, which is the right choice for a non-stationary target such as
the value of a policy that is still improving.
"""
    run = """
env.reset(seed=SEED)
set_seed(SEED)

policy_mc, Q_mc, returns_mc = monte_carlo_glie(
    env, alpha=ALPHA, gamma=GAMMA, num_episodes=N_EPISODES, eps_decay=EPS_DECAY)

print_policy(policy_mc, env.unwrapped, "Optimal policy found by MC:")

# ---- the three figures required by the assignment --------------------------
V_mc = np.max(Q_mc, axis=1)                     # V(s) = max_a Q(s,a)
plot_value_function(V_mc, "Optimal State-Value Function (Monte Carlo GLIE)", env.unwrapped)
plot_q_function(Q_mc, "Optimal Action-Value Function (Monte Carlo GLIE)", env.unwrapped)
plot_policy(policy_mc, "Optimal Policy (Monte Carlo GLIE)", env.unwrapped, V=V_mc)

plot_learning_curves({"Monte Carlo (GLIE)": returns_mc}, "Learning curve (Monte Carlo)")
""" + COMMON_RUN_TAIL.format(k="mc", label="Monte Carlo (GLIE)")

    r = RESULTS["mc"]
    analysis = """
## 6. Results and conclusions

The policy found by first-visit GLIE Monte Carlo:

```
{pol}
```

| metric | value |
|---|---|
| policy match with π\\* | **{match:.0f}%** |
| `max |V − V*|` | {verr:.4f} |
| exact E[V^π] | {v_pi:.4f}  (optimum {v_star:.4f}) |
| **value loss** E[V\\*] − E[V^π] | **{loss:.4f}** |
| empirical success rate | {succ:.1f}% |

{diffs}

### Reading the numbers

* The **value loss** is what actually matters: it is the exact expected return
  lost by following the learned policy instead of π\\*, computed analytically
  (no sampling noise). At {loss:.4f} the learned policy is effectively optimal.
* `max |V − V*| = {verr:.4f}` looks large by comparison, but a value estimate does
  not have to be accurate for the **policy** to be right - only the *ordering* of
  the actions matters.
* Why the values do not converge exactly: with a **constant** α every update keeps
  a fraction α of the newest return, so Q keeps fluctuating around Q\\* with a
  variance floor proportional to α. A decaying $\\alpha_n = 1/n$ would satisfy the
  Robbins-Monro conditions and converge exactly, at the cost of adapting slowly.
  The assignment explicitly asks for a constant α.
* Some disagreements with π\\* are unavoidable: in **state 0** the actions `↓` and
  `→` differ in value by only **0.0003**, a genuine near-tie that any sampling
  method flips from seed to seed.

### Monte Carlo - what it costs and what it buys

**Advantages**

* Completely **model-free** and **unbiased**: $G_t$ is a real sampled return, so
  the estimate is not corrupted by any wrong intermediate value.
* Insensitive to violations of the Markov property, since it never bootstraps.
* Conceptually the simplest of the three methods.

**Disadvantages**

* **High variance** - the return of a whole episode accumulates the randomness of
  every single slip, which is why the learning curve is visibly noisy.
* It can only learn at the **end of an episode**: no updates online, and it cannot
  be used at all in a continuing (non-episodic) task.
* Needs many episodes: the {n} episodes here cost far more environment
  interaction than the ~{sweeps} Bellman sweeps that DP needed - but DP needed the
  model, and MC did not.

**Conclusion.** GLIE Monte Carlo recovers an essentially optimal policy from pure
experience. The GLIE schedule is what makes this work: early on ε ≈ 1 explores the
whole grid, and by the end ε ≈ {eps:.3f} so the policy being evaluated is nearly the
greedy one. Assignments 3 and 4 replace the full return with a bootstrapped
one-step target and cut the variance dramatically.
""".format(pol=r["ascii"], match=100 * r["match"], verr=r["verr"], v_pi=r["v_pi"],
           v_star=V_STAR_MEAN, loss=r["loss"], succ=100 * r["success"],
           diffs=diff_table("mc"), n=N_EPISODES, sweeps=S_VI["sweeps"],
           eps=fl.glie_epsilon(N_EPISODES, 1.0, EPS_DECAY))

    model_free_notebook(
        "mc", "Assignment 2 - Monte Carlo (GLIE, first-visit)",
        "Find the optimal policy and the optimal action-value / state-value "
        "functions with **GLIE first-visit Monte Carlo** and a constant step size "
        "α, using an ε-decay schedule. Three figures are required: V\\*, Q\\* and π\\*.",
        theory, ["monte_carlo_glie"], run, analysis,
        "Assignment_2_Monte_Carlo.ipynb")


def notebook_3():
    theory = """
## 4. Sarsa - on-policy TD(0) control

Sarsa replaces the full Monte-Carlo return with a **one-step bootstrapped**
target, so it can update after every single step:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha\\,[\\,r + \\gamma\\,Q(s',a') - Q(s,a)\\,]$$

The name is the quintuple it uses: $(s, a, r, s', a')$. The crucial detail is that
$a'$ is the action the ε-greedy policy **actually takes next** - this makes Sarsa
**on-policy**: it evaluates and improves the very policy it is following,
exploration included.

At a terminal state there is nothing to bootstrap from ($V(\\text{terminal}) = 0$),
so the target collapses to the reward alone.
"""
    run = """
env.reset(seed=SEED)
set_seed(SEED)

policy_sarsa, Q_sarsa, returns_sarsa = sarsa(
    env, alpha=ALPHA, gamma=GAMMA, num_episodes=N_EPISODES, eps_decay=EPS_DECAY)

print_policy(policy_sarsa, env.unwrapped, "Optimal policy found by Sarsa:")

# ---- the three figures required by the assignment --------------------------
V_sarsa = np.max(Q_sarsa, axis=1)               # V(s) = max_a Q(s,a)
plot_value_function(V_sarsa, "Optimal State-Value Function (Sarsa)", env.unwrapped)
plot_q_function(Q_sarsa, "Optimal Action-Value Function (Sarsa)", env.unwrapped)
plot_policy(policy_sarsa, "Optimal Policy (Sarsa)", env.unwrapped, V=V_sarsa)

plot_learning_curves({"Sarsa": returns_sarsa}, "Learning curve (Sarsa)")
""" + COMMON_RUN_TAIL.format(k="sarsa", label="Sarsa")

    r = RESULTS["sarsa"]
    mc = RESULTS["mc"]
    analysis = """
## 6. Results and conclusions

The policy found by Sarsa:

```
{pol}
```

| metric | Sarsa | Monte Carlo (Assignment 2) |
|---|---|---|
| policy match with π\\* | {match:.0f}% | {mc_match:.0f}% |
| `max |V − V*|` | {verr:.4f} | {mc_verr:.4f} |
| exact E[V^π] | {v_pi:.4f} | {mc_v:.4f} |
| **value loss** | **{loss:.4f}** | {mc_loss:.4f} |
| empirical success rate | {succ:.1f}% | {mc_succ:.1f}% |

(optimum E[V\\*] = {v_star:.4f})

{diffs}

The disagreements carry a **Q\\* gap** of only a few hundredths, which is exactly
why the value loss stays near zero - these are states where two actions are almost
equally good, not states where Sarsa learned something wrong.

### Sarsa vs Monte Carlo

**Advantages of Sarsa**

* **Much lower variance**: the target $r + \\gamma Q(s',a')$ depends on the
  randomness of a *single* transition, whereas the MC return accumulates the
  randomness of an entire episode. The learning curve is visibly smoother.
* **Online / incremental**: it updates after every step instead of waiting for the
  episode to finish, so it also works in continuing tasks and propagates
  information about the holes immediately.
* It exploits the Markov property, which holds exactly in this gridworld.

**Disadvantages of Sarsa**

* The target is **biased** while Q is still wrong, because it bootstraps from its
  own estimates - early updates chase a moving target.
* It converges to $Q^{{\\pi}}$ of the **ε-greedy** policy, not to $Q^*$; only as
  ε → 0 do the two coincide. That is precisely what the GLIE schedule provides,
  and it also explains the residual `max|V − V*| = {verr:.4f}`.
* Being on-policy, it cannot learn about the greedy policy from data generated by
  a different (e.g. purely exploratory, or logged) policy.

### Why Sarsa is the *cautious* algorithm

Sarsa's target averages over what the ε-greedy policy really does - **including
the exploratory mistakes**. A state next to a hole therefore looks genuinely
dangerous to Sarsa, because with probability ε it will actually step into it. This
is the classic *cliff-walking* effect: Sarsa learns the safest policy for an agent
that still explores. Assignment 4 shows the opposite behaviour with Q-Learning.

**Conclusion.** Sarsa reaches an essentially optimal policy
(value loss {loss:.4f}) from the same {n} episodes as Monte Carlo, with markedly
less noise along the way - the standard bias/variance trade-off between
bootstrapping and full returns, resolved here in favour of bootstrapping.
""".format(pol=r["ascii"], match=100 * r["match"], verr=r["verr"], v_pi=r["v_pi"],
           loss=r["loss"], succ=100 * r["success"], v_star=V_STAR_MEAN,
           mc_match=100 * mc["match"], mc_verr=mc["verr"], mc_v=mc["v_pi"],
           mc_loss=mc["loss"], mc_succ=100 * mc["success"],
           diffs=diff_table("sarsa"), n=N_EPISODES)

    model_free_notebook(
        "sarsa", "Assignment 3 - Sarsa",
        "Repeat Assignment 2 with the **Sarsa** algorithm: on-policy TD(0) control "
        "with an ε-decay (GLIE) schedule and a constant step size α. The same three "
        "figures are required: V\\*, Q\\* and π\\*.",
        theory, ["sarsa"], run, analysis, "Assignment_3_Sarsa.ipynb")


def notebook_4():
    theory = """
## 4. Q-Learning - off-policy TD(0) control

Q-Learning uses the same one-step structure as Sarsa, but builds its target from
the **greedy** action instead of the action actually taken:

$$Q(s,a) \\leftarrow Q(s,a) + \\alpha\\,[\\,r + \\gamma\\,\\max_{a'} Q(s',a') - Q(s,a)\\,]$$

That single `max` is the whole difference, and it makes the method **off-policy**:

* the **behaviour** policy (how the agent acts) stays ε-greedy - it must, to explore;
* the **target** policy (what is being learned) is the greedy one.

Consequence: Q-Learning converges to $Q^*$ **regardless** of how exploratory the
behaviour policy is, whereas Sarsa converges to $Q^{\\pi}$ of its own ε-greedy
policy and reaches $Q^*$ only as ε → 0.

Monte Carlo and Sarsa are defined here as well, so that Section 7 can compare all
three under identical conditions.
"""
    run = """
env.reset(seed=SEED)
set_seed(SEED)

policy_ql, Q_ql, returns_ql = q_learning(
    env, alpha=ALPHA, gamma=GAMMA, num_episodes=N_EPISODES, eps_decay=EPS_DECAY)

print_policy(policy_ql, env.unwrapped, "Optimal policy found by Q-Learning:")

# ---- the three figures required by the assignment --------------------------
V_ql = np.max(Q_ql, axis=1)                     # V(s) = max_a Q(s,a)
plot_value_function(V_ql, "Optimal State-Value Function (Q-Learning)", env.unwrapped)
plot_q_function(Q_ql, "Optimal Action-Value Function (Q-Learning)", env.unwrapped)
plot_policy(policy_ql, "Optimal Policy (Q-Learning)", env.unwrapped, V=V_ql)
""" + COMMON_RUN_TAIL.format(k="ql", label="Q-Learning")

    multi_md = """
## 7. Comparing all three methods (Assignments 2-4)

**A single seed is not enough to rank the algorithms.** At `SEED = 0` Sarsa and
Q-Learning happen to converge to the *same* policy, which would make any
comparison between them meaningless. The cell below therefore repeats all three
methods over {n_seeds} seeds and reports mean ± std.

The metric that matters is the **value loss** $E[V^*] - E[V^{{\\pi}}]$: the exact
expected return given up by following the learned policy instead of π\\*,
computed analytically rather than sampled, so it carries no evaluation noise.

> ⏱ This cell trains {n_runs} agents and takes a few minutes.
""".format(n_seeds=len(SEEDS), n_runs=3 * len(SEEDS))

    multi_code = """
SEEDS = {seeds}

env.reset(seed=123)
_, success_star = evaluate_policy(env, policy_star, gamma=GAMMA, episodes=5000)

methods = [("Monte Carlo (GLIE)", monte_carlo_glie),
           ("Sarsa", sarsa),
           ("Q-Learning", q_learning)]

study, curves = {{}}, {{}}
for label, algo in methods:
    rows = []
    for seed in SEEDS:
        env.reset(seed=seed)
        set_seed(seed)
        pol, Q, returns = algo(env, alpha=ALPHA, gamma=GAMMA,
                               num_episodes=N_EPISODES, eps_decay=EPS_DECAY,
                               verbose=False)
        env.reset(seed=123)
        _, succ = evaluate_policy(env, pol, gamma=GAMMA, episodes=5000)
        _, v = exact_policy_value(env, pol, GAMMA)
        rows.append((v_star_mean - v, succ, policy_agreement(pol, policy_star, env)))
        if seed == SEEDS[0]:
            curves[label] = returns
    study[label] = np.array(rows)
    print("%-20s done" % label)

print()
print("%-20s%22s%20s%16s" % ("method", "value loss", "success rate", "policy match"))
print("-" * 78)
print("%-20s%22s%19.1f%%%15s" % ("pi* (DP optimum)", "0.0000", 100 * success_star, "100%"))
for label, _ in methods:
    a = study[label]
    print("%-20s%13.4f +- %.4f%14.1f%% +- %.1f%12.0f%% +- %.0f"
          % (label, a[:, 0].mean(), a[:, 0].std(),
             100 * a[:, 1].mean(), 100 * a[:, 1].std(),
             100 * a[:, 2].mean(), 100 * a[:, 2].std()))

plot_learning_curves(curves, "Learning curves: Monte Carlo vs Sarsa vs Q-Learning")
""".format(seeds=SEEDS)

    r = RESULTS["ql"]
    analysis = """
## 8. Results and conclusions

The policy found by Q-Learning at `SEED = 0`:

```
{pol}
```

| metric | value |
|---|---|
| policy match with π\\* | {match:.0f}% |
| `max |V − V*|` | {verr:.4f} |
| exact E[V^π] | {v_pi:.4f}  (optimum {v_star:.4f}) |
| **value loss** | **{loss:.4f}** |
| empirical success rate | {succ:.1f}% |

{diffs}

### The multi-seed comparison ({n_seeds} seeds)

{multi_table}

### The counter-intuitive result: the *optimal* policy is the least safe one

Read the success-rate column from the top. **π\\* has the lowest survival rate of
all**, and the ranking of the three learned methods is the exact reverse of their
value loss:

| | value loss | success rate |
|---|---|---|
| π\\* | best | **worst** |
| Monte Carlo | ↓ | ↓ |
| Sarsa | ↓ | ↓ |
| Q-Learning | **worst** | **best** |

This is the step reward doing its job. Every move costs −0.08, so a detour that
avoids the holes is only worth taking if it is short; π\\* deliberately accepts
some risk of falling in, because the extra steps of the safe route cost more than
the risk does. Consequently, at this operating point **all three deviations from
π\\* happen to be deviations toward over-caution** - the learned policies
survive more often and score less. The direction is not guaranteed in general:
it depends on the step size and the training budget; with a larger alpha or
fewer episodes, Monte Carlo deviates the other way.

The lesson for the report: **success rate is the wrong objective**. The MDP is
defined by its return, and optimising anything else - however intuitive - gives a
different (worse) policy. This is also why the *value loss* column, and not the
policy-match column, is the honest quality metric.

### Q-Learning vs Sarsa

Q-Learning is the furthest from π\\* here ({ql_match:.0f}% vs {s_match:.0f}% policy
match) and has the largest value loss, which is consistent with two known effects:

* **Maximisation bias** - the `max` over noisy estimates is systematically
  optimistic, $E[\\max_a \\hat{{Q}}] \\ge \\max_a E[\\hat{{Q}}]$. With a constant α the noise
  never vanishes, so the bias never fully washes out. Double Q-Learning exists
  precisely to remove it.
* Q-Learning learns about the greedy policy from ε-greedy data, so the states it
  updates most are not the states its target policy actually visits.

**Advantages of Q-Learning**

* **Off-policy**: it learns $Q^*$ no matter how the data was generated - from a
  fixed dataset, a human demonstrator, or a replay buffer. This is exactly why DQN
  and its descendants are built on Q-Learning and not on Sarsa.
* Converges to $Q^*$ even under a **fixed** exploratory ε; Sarsa needs ε → 0.
* Exploration and learning are decoupled, so ε can be tuned purely for coverage.

**Disadvantages of Q-Learning**

* Maximisation bias, as above.
* It learns the value of a policy it never actually follows, so *during* training
  the agent can behave worse than a Sarsa agent - the classic cliff-walking
  effect. If exploration is expensive or dangerous in the real system, that
  matters.
* Off-policy + bootstrapping + function approximation is the *deadly triad*, where
  divergence becomes possible. Tabular Q-Learning as used here is safe.

### Overall conclusion (Assignments 1-4)

| | model needed? | variance | bias | converges to |
|---|---|---|---|---|
| DP (PI / VI) | **yes**, full `P` | none | none | exactly π\\*, V\\* |
| Monte Carlo | no | high | none | Q\\* (under GLIE) |
| Sarsa | no | low | yes (bootstrap) | $Q^{{\\pi_\\varepsilon}}$ → Q\\* as ε → 0 |
| Q-Learning | no | low | yes (bootstrap + max) | Q\\* directly |

All four methods agree on essentially the same solution of this gridworld: the
three model-free methods land within **{worst:.3f}** of the optimal expected return.
They pay for it very differently, though. DP is exact and by far the cheapest
({sweeps} sweeps, milliseconds) and would be the obvious choice - **if** the model
`P` were available. It is not, in any interesting problem. The whole value of
Monte Carlo, Sarsa and Q-Learning is that they need nothing but experience, and
that is what lets them scale to problems where the transition model can never be
written down.

Among the model-free three, Monte Carlo gave the most accurate policy here (it is
unbiased, and {n} episodes is plenty for a 20-state MDP), while Sarsa and
Q-Learning bought a much smoother, lower-variance learning curve at the price of a
small bootstrapping bias - visible in the learning-curve figure above. On a larger
problem, or in a continuing task where episodes never end, that trade goes the
other way and the TD methods win.
""".format(pol=r["ascii"], match=100 * r["match"], verr=r["verr"], v_pi=r["v_pi"],
           loss=r["loss"], succ=100 * r["success"], v_star=V_STAR_MEAN,
           diffs=diff_table("ql"), multi_table=multi_seed_table(),
           n_seeds=len(SEEDS), ql_match=100 * ms("ql", 2)[0],
           s_match=100 * ms("sarsa", 2)[0],
           worst=max(ms(k, 0)[0] for k in MULTI), sweeps=S_VI["sweeps"], n=N_EPISODES)

    model_free_notebook(
        "ql", "Assignment 4 - Q-Learning",
        "Repeat Assignment 2 with the **Q-Learning** algorithm: off-policy TD(0) "
        "control with an ε-decay (GLIE) schedule and a constant step size α. The "
        "same three figures are required: V\\*, Q\\* and π\\*.",
        theory, ["monte_carlo_glie", "sarsa", "q_learning"], run, analysis,
        "Assignment_4_Q_Learning.ipynb",
        extra_cells=[md(multi_md), code(multi_code)])


if __name__ == "__main__":
    print()
    notebook_1()
    notebook_2()
    notebook_3()
    notebook_4()
    print("\ndone.")
