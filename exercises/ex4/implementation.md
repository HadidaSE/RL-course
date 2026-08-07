# Assignment 4 — What Was Implemented and How

This document explains every piece of the ex4 solution: the design decisions,
how each file works, and how the pieces fit together. For *how to run it*,
see `instructions.md`; for the submission-facing summary (hyperparameters,
observation-alternative justification, results), see `report.md`.

---

## 1. The problem in one paragraph

The Box Pushing world of Assignments 1–2 becomes a **POMDP**: the robot no
longer knows its own location. Everything else is known — the map, the box
positions, the stochastic dynamics (move 0.8/0.1/0.1, push 0.8/0.2). At each
step the robot receives only a 3×3 window of the board around its true (but
hidden) position. It must *localize itself while acting* and still push all
boxes onto the goals. We solve it with **POMCP** (Silver & Veness, 2010):
online Monte-Carlo tree search over a **particle-filter belief**, planning a
fresh decision every step under a hard wall-clock budget (1 s / 20 s).

## 2. Architecture overview

```
                 ┌──────────────────────────────┐
                 │  solution_ex4.py             │
                 │  online loop + experiments   │
                 └──────┬──────────────┬────────┘
        plan(belief)    │              │  step(action) / obs
                        ▼              ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │ pomcp.py         │   │ pomdp_env.py         │
        │ POMCP (MCTS+UCB1)│   │ REAL world: wraps    │
        └──────┬───────────┘   │ course stochastic env│
               │ simulate       └──────────┬───────────┘
               ▼                           │ real (a, o)
        ┌──────────────────┐               ▼
        │ pomdp_model.py   │◄──┌──────────────────────┐
        │ generative model │   │ particle_filter.py   │
        │ G(s,a)→(s',o,r)  │   │ belief over states   │
        └──────────────────┘   └──────────────────────┘
```

Two "worlds" exist on purpose:

* **The real world** (`pomdp_env.py`) is the course-provided
  `StochasticMultiAgentBoxPushEnv` — the ground truth the agent acts in.
* **The imagined world** (`pomdp_model.py`) is a fast, compact reimplementation
  of the *same* dynamics, used thousands of times per second by POMCP
  simulations and by the particle-filter update. POMCP is only correct if
  this model matches the real environment exactly — which is why
  `test_ex4.py` proves trajectory-level parity between the two.

## 3. `pomdp_model.py` — the generative model

**State** (`State` namedtuple, hashable):
`(agents, small, heavy)` — per-agent (x, y) positions plus sorted tuples of
small/heavy box positions. Box positions live *inside* each particle even
though they start out fully known, because whether a push succeeded depends
on the hidden agent location — after the first push attempt, box uncertainty
and location uncertainty are entangled.

**Actions are compass moves** (0=right, 1=down, 2=left, 3=up), matching the
assignment's note that moves are direct with no facing direction. The
MiniGrid rotations of the underlying env are deterministic and produce no
observation, so they are pure bookkeeping — the env wrapper absorbs them
(sets the facing direction, then issues one `forward`). One compass action
therefore equals exactly one environment step and one POMDP transition.

**Transition** (`step`) mirrors the course env pass-for-pass:

1. *Heavy-box joint pushes first*: ≥2 agents targeting the same heavy box
   from the same cell in the same direction → box moves with p=0.8; their
   intents are consumed whatever the outcome (exactly like the env).
2. *Individual actions in agent order*: small-box push (p=0.8 success, 0.2
   no-op), or move with directional noise (p=0.8 intended, 0.1 each 90°
   side; a deviation into a blocked cell leaves the agent in place), or
   no-op on unmet preconditions. Agents never block one another — the env
   clears agent sprites during movement resolution, and the model replicates
   that.

Reward is sparse and identical to previous assignments: 1.0 exactly when all
goal cells are covered by boxes (terminal), else 0. The code's γ default is
0.95 (matching previous assignments), but the CLI default — and every
reported experiment — uses **γ = 0.99** for a longer effective planning
horizon; see `report.md` §5, where the deviation is declared.

**Observation function — both alternatives implemented** (`window` /
`observe`, placement selected by `obs_mode` via the shared `OBS_OFFSETS`
table):

* `"egocentric"` (**Alternative B**, the default): a 3×3 window *centred* on
  the agent's true location.
* `"north"` (**Alternative A**): a 3×3 window lying *immediately north* of
  the agent — its bottom row is the row directly above the agent, columns
  x−1..x+1 — regardless of any facing direction. Near the top border parts
  of this window fall off the board and read as WALL.

Each cell is FREE / WALL / BOX; goal cells and other agents read as FREE,
keeping the window a deterministic function of the agent's own location plus
the box layout (O(o|s′)=1 iff o is exactly that window). We use B for the
report: symmetric information for all four move directions, never sticks out
of a wall-bordered board, and localizes faster (in smoke runs the north
window needed ~3× more steps on the two-robot map). Neither is MiniGrid's
built-in facing-forward view — both are our own slices of the full board, as
the assignment requires. Select with `--obs egocentric|north`.

## 4. `pomdp_env.py` — the real-world adapter

`BoxPushPOMDPEnv` wraps the course env and exposes the POMDP interface:

* `step({agent: direction})` sets each agent's facing direction (deterministic,
  information-free) and executes a single `forward` — one env step per
  compass action, so the real dynamics are literally the course env's.
* Observations are produced by our own `_window` that slices the true grid
  around each agent's hidden position with the same FREE/WALL/BOX encoding
  as the model (agents/goals read FREE).
* `randomize_start=True` teleports the agents after every reset to cells
  drawn uniformly from the free cells — i.e. the true initial state is a
  genuine sample from the uniform initial belief b₀. Without this, all runs
  would start from the map's fixed 'A' cells while the agent *believes* it
  could be anywhere; sampling b₀ is the faithful POMDP simulation. (Reported
  as a deviation in `report.md`.)
* `true_state()` exists **only** for tests/logging; the planner never reads it.

## 5. `particle_filter.py` — the belief state

The belief is a bag of `N = 500` **unweighted particles** (full `State`s), as
in the original POMCP paper.

* **Initialization**: location hypotheses spread uniformly over all free
  (non-wall, non-box) cells; every particle carries the known initial box
  layout. With two robots, each particle holds a *joint* location hypothesis.
* **Initial-observation conditioning** (`condition_on_observation`):
  `env.reset()` returns an observation before any action. Because the
  observation function is deterministic, exact Bayesian conditioning of the
  uniform belief is simply *keep the location hypotheses whose window
  matches*, refilled to N. This is a small, principled extension of the
  assignment's pseudocode (which ignores the reset observation).
* **Update** (`update`) — the assignment-mandated rejection sampling: sample
  a particle, simulate the *executed* action through the same generative
  model, keep the successor iff its simulated observation equals the real
  one; repeat until N particles are collected, capped at `100·N` attempts.
* **Depletion handling / reinvigoration** (`_reinvigorate`): deterministic
  observations make the rejection rate potentially high; if the cap is hit
  the filter is refilled with **exactly-consistent states**, leveraging full
  map knowledge (the option the assignment explicitly suggests): for each
  box configuration present in the surviving belief (or one-step simulations
  of the old belief if nothing survived), enumerate every free cell whose
  3×3 window matches the real observation per agent, and sample states from
  that set. Co-located agents are allowed (needed for heavy-box pushes).
  Two further fallbacks (duplicate survivors → uniform re-init) make the
  filter impossible to empty.
* **Measured behaviour**: instrumented over 178 belief updates
  (`--verbose --jobs 1`), rejection sampling refilled all 500 particles
  every single time and reinvigoration never fired — mean acceptance rate
  19.1%, but only 1.0% at the worst step (48,804 of 50,000 permitted
  attempts). The fallback is therefore unexercised in practice yet not
  redundant; see `report.md` §5.

## 6. `pomcp.py` — the planner

`POMCPPlanner.plan(particles, time_budget)` runs simulations from the current
belief until the deadline:

* **Root sampling**: each simulation starts from a state drawn uniformly
  from the particle filter — POMCP operates directly on the particle belief,
  never an explicit distribution.
* **Tree policy (UCB1)**: inside the tree, action `a` maximizes
  `Q(ha) + c·sqrt(log N(h) / N(ha))` with `c = 1.0`; untried actions first.
  Tree nodes are histories; children are keyed by (action, observation), so
  identical observation sequences share statistics.
* **Tree reuse across decisions**: the tree is *not* rebuilt every step. After
  the real action `a` and observation `o`, `advance(a, o)` prunes it to the
  `T(hao)` subtree, so the statistics gathered by earlier simulations are
  carried into the next decision — the canonical Silver & Veness (2010)
  formulation. `reset()` clears it at the start of each episode, and if the
  real observation was never simulated (no matching child) the tree is simply
  dropped and the next `plan()` starts fresh.
* **Preferred actions**: on expansion, the rollout policy's greedy joint
  action is recorded in `_Node.preferred` and tried first among the untried
  actions, warm-starting the 16-arm joint search instead of probing arms in
  random order.
* **Expansion + rollout**: the first time a simulation leaves the tree, one
  node is added and the value below it is estimated by a depth-limited
  rollout (horizon `max_depth = 60`).
* **Rollout policy** (`HeuristicRolloutPolicy`): greedy-with-noise — ε = 0.2
  fully random; otherwise each agent prefers pushes that bring a box closer
  to an uncovered goal, then moves that reduce Manhattan distance to *its
  assigned* unfinished box; wall bumps and deadlock-prone pushes are
  penalized. With two robots, `_assign()` first greedily matches each agent
  to a **distinct** unfinished box (nearest-first, hand-rolled — no `scipy`)
  so the robots do not chase the same one. The assignment allows "a random or
  simple heuristic policy"; a purely random rollout essentially never reaches
  the sparse terminal reward within the horizon, starving the search of
  signal at 1 s budgets.
* **Backup**: returns are propagated up, updating visit counts and running
  means of every (history, action) edge on the path.
* **Budget enforcement**: the `time.monotonic()` deadline is checked in the
  top-level loop *and* at every tree depth, so the decision truly returns
  within the budget (unit-tested). Action choice at the root = highest
  estimated value among visited actions, per the assignment.
* **Multi-agent**: planning is centralized — joint actions (4² = 16 arms at
  each node) over joint location hypotheses.
* **No privileged information**: the rollout heuristic, the agent→box
  matching and the preferred-action choice all read a state *sampled from the
  belief* plus the known map/box layout — never the true hidden location. The
  executed action is the root action, i.e. an average over the whole belief.

## 7. `solution_ex4.py` — online loop and experiments

`run_episode` is the assignment's loop verbatim:

```python
obs, _ = env.reset()
pf.initialize(); pf.condition_on_observation(obs)   # uniform b0 ∩ first obs
planner.reset()                                      # clear retained tree
while not done:
    action = planner.plan(pf.particles, time_budget) # 1. plan (POMCP)
    obs, r, term, trunc, _ = env.step(action)        # 2. act in real env
    pf.update(action, obs)                           # 3. belief update
    planner.advance(action, obs)                     #    prune tree to T(hao)
```

`run_experiment` repeats it `n_runs = 30` times per (scenario, budget) cell
with per-run seeds (`base_seed + 10000·i`), optionally in parallel
(`--jobs`, one episode per process), and aggregates mean/std steps and solve
rate. Truncated episodes (200-step cap) count as failures and are **excluded
from the reported mean/std** — a run that never solved has no meaningful
"steps to solve", and at the cap a single one would distort a 30-run cell
badly. The solve rate reports how many were excluded, and the JSON keeps both
views (`mean_steps` / `std_steps` over solved runs, `mean_steps_all` /
`std_steps_all` over all runs). Results go to a log file (console mirrors
INFO; DEBUG keeps per-step particle/simulation diagnostics) and a raw JSON.

**Scenarios** (`MAPS`): `single` (1 robot, 1 box, 1 goal) and `multi`
(2 robots, 2 boxes, 2 goals) on 7×6 boards, plus the full Assignment-1/2
heavy-box map as `ex2` (supported and parity-tested, but off by default —
a heavy box needs both robots pushing *from the same cell simultaneously*,
which under dual location uncertainty makes episodes measure a coordination
lottery rather than planning quality; reported as a deviation).

## 8. `test_ex4.py` — how correctness is established

| Test | What it proves |
|------|----------------|
| `test_model_matches_env_deterministically` | With success probs forced to 1.0 both worlds are deterministic; on all three maps (incl. heavy-box ex2) the model reproduces the real env's states **and** observations action-for-action for 300 steps. This is the linchpin: it means the particle filter's generative model is exact. |
| `test_observation_function_matches_under_stochastic_dynamics` | Under normal stochastic dynamics, the wrapper's real-grid window always equals the model's window for the true state. |
| `test_particle_filter_tracks_true_state` | Along a random-action episode, the true hidden state stays inside the 300-particle belief in >90 % of steps. |
| `test_particle_filter_survives_depletion` | An impossible observation triggers reinvigoration and still yields N particles. |
| `test_north_window_semantics` | Alternative A's window really sits immediately north of the agent (bottom row = the row above it, columns x−1..x+1) and reads off-board cells as WALL. |
| `test_pomcp_budget_is_enforced` | `plan()` with a 0.5 s budget returns within tolerance and completes >100 simulations. |
| `test_online_loop_solves_single_scenario` | A full episode with a 0.3 s budget actually solves the single-robot map. |

All seven pass; a `--quick` smoke run solves both default scenarios with 100 %
success even at a 0.1 s budget.

## 9. Design decisions at a glance

| Decision | Choice | Why |
|---|---|---|
| Observation | Alternative B (egocentric 3×3) by default; Alternative A ("north") also implemented, `--obs` selects | symmetric info for direct moves; always in-bounds; faster localization (A kept for comparison) |
| Action space | 4 compass directions (joint for 2 robots) | matches assignment's direct-move semantics; rotations are information-free bookkeeping |
| Belief | 500 unweighted particles, full states | original POMCP representation; boxes inside particles because push outcomes depend on hidden location |
| Belief update | rejection sampling + map-based reinvigoration | assignment-mandated; deterministic obs ⇒ depletion must be handled |
| Rollouts | greedy-with-noise heuristic (ε = 0.2) with agent→box task allocation | sparse reward starves random rollouts at 1 s budgets; distinct box assignment stops two robots chasing the same box |
| Horizon | 60 | ≈3–5× the typical solve length. At the code default γ = 0.95, γ⁶⁰ ≈ 0.046 so deeper reward is negligible; at the reported γ = 0.99, γ⁶⁰ ≈ 0.547, so the horizon is set by solve length rather than by the discount tail |
| UCB c | 1.0 | recommended start; Q ∈ [0,1] here so c = 1 explores adequately |
| Tree | retained across decisions, pruned to `T(hao)` | canonical POMCP; reuses earlier simulations instead of discarding them each step |
| Node expansion | preferred (greedy) action tried first | warm-starts the 16-arm joint search under a tight budget |
| Starts | randomized uniformly per reset | the true initial state should be a sample from b₀ |

## 10. Bug found and fixed: simultaneous agent/box collisions

While debugging a 20 s-budget `multi` run that never progressed for ~66
minutes (200 steps × 20 s), we found that `environment/stochastic_env.py`'s
`StochasticMultiAgentBoxPushEnv.step()` resolved each agent's move/push
**one at a time**, checking a destination cell only against static grid
content. It never checked whether a *different* agent's simultaneous move,
or a box a different agent pushed in the same step, would land on the exact
same cell. When that happened, the later "regenerate observations" loop
(`grid.set(*pos, None)` per agent, before placing its sprite) silently
**deleted** whichever box had been pushed onto that agent's landing cell —
nothing else in the codebase tracks box positions independently of the
grid. One goal then became permanently uncoverable, and the affected
episode ran to `max_steps` without ever raising an error — exactly the
symptom observed.

**Fix** (`environment/stochastic_env.py`, Pass 3 of `step()`): every agent's
outcome is now planned against a frozen grid snapshot (no incremental
mutation while planning — this matches the "simultaneous joint action"
semantics already used elsewhere, e.g. the heavy-box joint push). Each plan
records which cells it claims (an agent's own landing cell, plus a pushed
box's landing cell); any cell claimed by more than one plan cancels every
plan involved, and only collision-free plans are applied. This also
naturally covers two agents pushing the *same* small box in the same step.
The original buggy code is kept as an inline comment directly above the fix
for reference. Regression tests: `tests/test_stochastic_env.py::TestSimultaneousCollisions`
(verified to fail against the pre-fix code, confirming they catch the bug).

**Consequence for this assignment**: `pomdp_model.py`'s generative model
(`BoxPushModel.step`, Pass 2) had the *identical* latent bug — since POMCP's
correctness depends on the model matching the real environment exactly
(enforced by `test_model_matches_env_deterministically`), fixing only the
real environment would have made the two silently disagree on collision
cases. The same fix (frozen-snapshot planning + claim-collision
cancellation) was mirrored there, with the same inline
original/bug-explanation/fix comment structure, so model and environment
stay in exact agreement.

This bug is **not specific to ex4** — it lives in shared course
infrastructure (`environment/stochastic_env.py`) also used by Assignment 2,
so any 2+-agent stochastic scenario there could hit it too, just less
consequentially (shorter episodes, no 20 s-per-decision multiplier).

## 11. Results at a glance

Full 30-run sweep, mean/std over solved runs only (see `report.md` §7 for the
discussion):

| Scenario | Budget | Mean steps | Std | Solve rate |
|---|---|---|---|---|
| single | 1 s | 6.70 | 3.29 | 100% (30/30) |
| single | 20 s | 6.63 | 3.19 | 100% (30/30) |
| multi | 1 s | 13.83 | 8.20 | 100% (30/30) |
| multi | 20 s | 9.28 | 3.08 | 97% (29/30) |

The single-robot map is never compute-starved, so the budget changes nothing
(6.70 → 6.63). The two-robot map is, so the larger budget cuts ~33% of the
steps (13.83 → 9.28) and tightens the spread (8.20 → 3.08) — the multi-agent
penalty shrinks from ≈2.1× to ≈1.4× the single-robot cost.

## 12. What is intentionally NOT here

* No POMDP/planning/RL libraries — POMCP, the particle filter and the
  observation function are implemented from scratch, per the assignment.
* No peeking: the planner and filter only ever see actions and observations;
  `true_state()` is used exclusively by tests and DEBUG logging.
* No bit-reproducibility. Seeds fix the environment's stochastic dynamics
  and the start location, but *not* how many POMCP simulations fit inside a
  wall-clock budget — that depends on machine load. Cells sensitive to a rare
  truncation therefore vary between executions of the same command; see
  `report.md` §7.
