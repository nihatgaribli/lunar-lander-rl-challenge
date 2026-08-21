# Frozen Lake 4x5 - Assignments 1-4 (Summer Semester 2026)

**Name:** `Nihat Garibli` - **ID:** `5425819`

Run everything with:

```bash
python frozen_lake_rl.py           # opens the plots
RL_SHOW_PLOTS=0 python frozen_lake_rl.py   # head-less, only saves to ./plots
```

Requirements: `gymnasium`, `numpy`, `matplotlib`, `seaborn`.

## 1. The environment

```
 S  F  F  F  F        states 0..4
 F  F  F  H  F        states 5..9    (hole at state 8)
 F  F  F  F  F        states 10..14
 H  F  F  F  G        states 15..19  (hole at 15, goal at 19)
```

| Setting | Value |
|---|---|
| Step reward (every transition) | −0.08 |
| Goal / hole bonus | +1 / −1 (added on top of the step reward) |
| P(intended direction) | 0.8 |
| P(slip to each perpendicular side) | 0.1 + 0.1 |
| Walls | rigid - the agent stays in place |
| Discount γ | 0.95 |
| Start state | uniformly random over the 17 non-terminal squares |

The step cost is what makes the problem interesting: waiting is expensive, so
the optimal policy is a compromise between the *shortest* path and the *safest*
path around the two holes.

Actions: `0 = ← (W)`, `1 = ↓ (S)`, `2 = → (E)`, `3 = ↑ (N)`.

## 2. Assignment 1 - Dynamic Programming

Both methods converge to the **same** optimal policy and the same V\* (`max|V_PI − V_VI| ≈ 5e-12`):

```
↓  ↓  ↓  →  ↓
→  ↓  ↓  H  ↓
→  →  ↓  ↓  ↓
H  →  →  →  G
```

| | Policy Iteration | Value Iteration |
|---|---|---|
| Outer iterations | 7 policy improvements | 28 sweeps |
| Total Bellman sweeps | 436 | 28 |
| Runtime | ~11 ms | ~3 ms |

**Comparison.**

* *Policy Iteration* needs very few **outer** iterations, because each improvement
  step is a large jump in policy space, and it terminates **exactly** (once the
  greedy policy stops changing, it is provably optimal - no ε threshold needed).
  The cost is that every iteration runs a full policy evaluation to convergence,
  which here means 436 sweeps in total.
* *Value Iteration* does only one Bellman **optimality** backup per sweep, so each
  sweep is cheap and no inner loop is needed. It converges asymptotically, so it
  stops on a threshold θ, and the number of sweeps grows as γ → 1
  (`≈ log(θ)/log(γ)`).
* On this tiny 20-state MDP, VI wins on wall-clock time. PI becomes attractive when
  evaluation can be solved directly (a 20×20 linear system here) or when γ is very
  close to 1, where VI needs many sweeps. In practice *modified* policy iteration
  (a truncated evaluation of k sweeps) interpolates between the two.

Notice the row-1 column-4 square: the policy goes `↓` rather than `←`, because
sliding left risks entering the hole at state 8. This is the slip probability
being priced into the policy.

## 3. Assignments 2-4 - model-free control

Identical settings for all three, so the comparison is fair:
constant `α = 0.02`, `γ = 0.95`, `200 000` episodes,
GLIE schedule `ε_k = 1/(1 + 2e-4·k)` (ε: 1.0 → 0.024).

`ε_k → 0` guarantees the policy becomes greedy in the limit, while the decay is slow
enough that every state-action pair keeps being visited - the two GLIE conditions.
Random start states help here too: they guarantee coverage of squares a greedy
policy would never revisit.

| method | max\|V−V\*\| | policy match | E[V^π] | value loss | success | time |
|---|---|---|---|---|---|---|
| Policy Iteration | 0.0000 | 100% | 0.4093 | 0.0000 | 95.4% | 0.01 s |
| Value Iteration | 0.0000 | 100% | 0.4093 | 0.0000 | 95.1% | 0.00 s |
| Monte Carlo (GLIE) | 0.1344 | 100% | 0.4093 | 0.0000 | 95.3% | 5.7 s |
| Sarsa | 0.1157 | 94% | 0.4036 | 0.0057 | 95.5% | 5.3 s |
| Q-Learning | 0.1160 | 88% | 0.3919 | 0.0174 | 97.3% | 7.2 s |

* **E[V^π]** is the *exact* expected discounted return of the learned policy
  (analytic policy evaluation, averaged over the random start states) - unlike a
  roll-out estimate it has no sampling noise, so identical policies always score
  identically.
* **value loss** = E[V\*] − E[V^π]. All three model-free methods end up within
  **0.02** of the optimum, i.e. essentially optimal policies.

### Why the value estimates stay ~0.12 away from V\*

With a **constant** step size the estimates never fully converge: each update
keeps a fraction α of the newest return, so Q keeps fluctuating around Q\* with a
variance floor proportional to α. This is deliberate - a constant α is what the
assignment asks for, and it is what you want in a non-stationary problem. A
decaying `α_n = 1/n` would satisfy the Robbins-Monro conditions and converge
exactly, at the price of adapting more slowly.

Note that a large `max|V−V*|` does **not** mean a bad policy: what matters for the
policy is the *ordering* of the actions, not the absolute values.

### Where the policies differ from π\*

```
Monte Carlo (GLIE)  matches pi* in every state.
Sarsa       state 10 (2,0):  pi* = →   learned = ↑   Q* gap = 0.0637
Q-Learning  state  7 (1,2):  pi* = ↓   learned = ←   Q* gap = 0.0698
            state 10 (2,0):  pi* = →   learned = ↑   Q* gap = 0.0637
```

The gaps are small in absolute terms, which is why the value loss stays near zero.
State 0 is an even more extreme case: `↓` and `→` differ by only **0.0003**, so any
sampling method flips between them from seed to seed - a genuine near-tie, not a
learning failure.

### Robustness: the same comparison over 5 seeds

A single seed is not enough to rank the methods - at `SEED = 0` Sarsa and
Q-Learning happen to find the *same* policy. Repeating all three over 5 seeds
(`Assignment_4_Q_Learning.ipynb`, section 7) gives:

| method | value loss | success rate | policy match |
|---|---|---|---|
| **pi\* (DP optimum)** | 0.0000 | **95.1%** | 100% |
| Monte Carlo (GLIE) | 0.0048 +/- 0.0071 | 95.5% +/- 0.3 | 95% +/- 3 |
| Sarsa | 0.0090 +/- 0.0029 | 96.2% +/- 0.5 | 92% +/- 3 |
| Q-Learning | 0.0138 +/- 0.0047 | 96.7% +/- 0.4 | 89% +/- 4 |

### The counter-intuitive result: the optimal policy is the least safe one

Read the success-rate column from the top: **pi\* has the lowest survival rate of
all**, and the ranking of the three learned methods is the exact reverse of their
value loss.

This is the step reward doing its job. Every move costs -0.08, so a detour around
a hole is only worth taking if it is short; pi\* deliberately accepts some risk of
falling in, because the extra steps of the safe route cost more than the risk
does. Consequently, at this operating point **all three deviations from pi\***
**happen to be deviations toward over-caution** - the learned policies survive
more often and score less. The direction is not guaranteed in general: it
depends on the step size and the training budget, and with a larger alpha or
fewer episodes Monte Carlo deviates the other way.

The lesson: **success rate is the wrong objective.** The MDP is defined by its
return, and optimising anything else - however intuitive - yields a different and
worse policy. That is also why *value loss*, not *policy match*, is the honest
quality metric here.

### Sarsa vs Q-Learning

* **Sarsa is on-policy**: its target uses the action actually taken next, so it
  learns the value of the ε-greedy policy it is following. Because that policy
  occasionally slips into a hole by accident, Sarsa is the more *conservative* of
  the two, and it converges to Q\* only as ε → 0.
* **Q-Learning is off-policy**: the `max` in its target evaluates the greedy policy
  regardless of how the agent explored, so it converges to Q\* even under a fixed
  exploratory ε.
* Their behaviour here shows exactly this trade-off: Q-Learning has the highest
  empirical **success rate** (97.3% vs 95.5%) - it prefers routes that stay further
  away from the holes - but a slightly **lower** E[V^π], because those routes are
  longer and every extra step costs −0.08. Higher survival, lower return.

### Monte Carlo vs TD

* MC uses the **complete** return G_t: unbiased, but high variance, and it can only
  learn at the end of an episode.
* Sarsa / Q-Learning **bootstrap** from the next state: much lower variance and
  online updates, at the price of bias while the estimates are still wrong.
* MC recovered the optimal policy exactly here; the price is that it needs
  the full episode and - visible in the learning curves - is noisier early on.

## 4. Files

Each assignment is a **standalone** notebook - it contains the environment, the
algorithm, the figures and the written analysis, and can be uploaded to Colab and
run on its own (as the submission requires, one assignment per box).

| File | Contents | Runtime |
|---|---|---|
| `Assignment_1_Dynamic_Programming.ipynb` | Policy Iteration + Value Iteration, PI vs VI comparison | ~5 s |
| `Assignment_2_Monte_Carlo.ipynb` | GLIE first-visit MC, constant alpha | ~10 s |
| `Assignment_3_Sarsa.ipynb` | Sarsa, on-policy TD(0) | ~10 s |
| `Assignment_4_Q_Learning.ipynb` | Q-Learning + the 5-seed comparison of all three methods | ~2-5 min |
| `frozen_lake_rl.py` | the same code as a single script that runs all four assignments | ~20 s |
| `build_notebooks.py` | regenerates the four notebooks from `frozen_lake_rl.py` | ~4 min |
| `plots/` | 14 figures produced by the script (V, Q and policy per method, plus learning curves) | |

The notebooks are **generated** from `frozen_lake_rl.py`, so the two can never
drift apart, and every number quoted in an analysis cell is measured by actually
running the algorithm rather than typed in by hand. To regenerate after changing
the implementation:

```bash
python build_notebooks.py
```

Key functions: `policy_iteration`, `value_iteration`, `monte_carlo_glie`,
`sarsa`, `q_learning`, `exact_policy_value`, `report_policy_differences`.

Author details (**Nihat Garibli**, ID **5425819**) are filled in at the top of
every notebook and of `frozen_lake_rl.py`.
