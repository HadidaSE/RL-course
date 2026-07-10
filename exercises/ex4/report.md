# Assignment 4 Report — POMDP Planning with POMCP in the Box Pushing Environment

Students: Itamar Hadida, Omer Aviram

---

## 1. Code Layout

| File | Contents |
|------|----------|
| `pomdp_model.py` | Generative model `G(s,a) → (s',o,r)` — compact state, the Assignment-2 stochastic dynamics, and the **observation function** (Alternative B). |
| `pomdp_env.py` | Wrapper around the course's real `StochasticMultiAgentBoxPushEnv`: compass actions, hidden agent location, and the same 3×3 egocentric observation sliced from the true grid. |
| `particle_filter.py` | **Particle filter**: uniform initialisation over free cells, unweighted rejection-sampling update (Silver & Veness 2010), and map-based particle reinvigoration for depletion. |
| `pomcp.py` | **POMCP**: particle-belief MCTS with UCB1 in-tree action selection, heuristic rollouts beyond the tree, and a wall-clock budget enforced inside the simulation loop. |
| `solution_ex4.py` | Online planning loop (plan → act → belief update) and the experiment harness / CLI. |
| `test_ex4.py` | Sanity tests, incl. exact trajectory parity between the generative model and the real environment. |

No POMDP / planning / RL libraries are used — POMCP, the particle filter and
the observation function are implemented from scratch.

Reproduce with:

```bash
python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4
python3 exercises/ex4/test_ex4.py            # sanity tests
```

## 2. Choice of Observation Alternative — B (egocentric)

We implemented **both** alternatives (selectable with `--obs egocentric|north`)
and use **Alternative B** for all reported experiments: a 3×3 window *centred*
on the agent's true (but hidden) location, where each cell reads FREE / WALL /
BOX (goal cells and other agents read as FREE). Alternative A ("north") places
the same 3×3 window immediately north of the agent (bottom row adjacent to the
agent's row), with off-board cells reading as WALL. Reasons for choosing B:

1. **Symmetric information for direct moves.** Our actions are direct compass
   moves (left/right/up/down, no facing direction), so information about all
   four neighbouring cells is equally valuable. Alternative A (a window fixed
   to the north) tells the agent nothing about the cell it is about to enter
   when moving south, east or west.
2. **Always in-bounds.** On wall-bordered maps an egocentric window around any
   reachable cell never leaves the board, whereas a "window to the north"
   sticks out over the border whenever the agent stands in the top row and
   needs an extra out-of-bounds convention.
3. **Faster localisation.** The centred window observes the walls/boxes the
   agent is adjacent to on every side, which prunes location hypotheses
   quickly (the belief usually collapses within a handful of steps). This is
   borne out empirically: in identical smoke runs the north window needed
   roughly 3× more steps on the two-robot map, since it carries no
   information about the cells east, west and south of the agent and is
   entirely blank (all-WALL) near the top border.

As required, the observation is **not** MiniGrid's built-in facing-forward
view: it is our own function that slices the window directly from the full
board representation around the robot's true location (both in the generative
model, `BoxPushModel.window`, and over the real grid, `BoxPushPOMDPEnv._window`).
Because there is no sensor noise, `O(o|s') = 1` iff `o` is exactly the window
around `s'`.

## 3. Implementation Notes

* **Actions.** One POMDP action = one compass move (or push, when a box is
  ahead). MiniGrid rotations are deterministic and observation-free, so the
  wrapper sets the facing direction and issues a single `forward`; one compass
  action = one environment step, and the transition model matches Assignment 2
  exactly (move 0.8/0.1/0.1, push 0.8/0.2, unmet precondition → no-op).
* **State / belief.** Each particle is a full world state (per-agent location
  + box layout). The *initial* uncertainty is only over the agents' own
  locations (boxes known), but particles must carry boxes too because a
  push's outcome depends on the hidden location. In the two-robot scenario
  planning is centralised: particles hold *joint* location hypotheses and
  POMCP searches over joint actions (4² = 16 arms).
* **Belief update.** Exactly the unweighted rejection-sampling update of the
  original POMCP: sample a particle, simulate the executed action through the
  same generative model, keep the successor iff the simulated observation
  equals the real one. Attempts are capped at `100·N`; on (partial) depletion
  we **reinvigorate** by exploiting full map knowledge — enumerating, per box
  configuration in the surviving belief, every free cell whose 3×3 window
  matches the real observation, and refilling the filter from that
  exactly-consistent set.
* **Initial observation.** `env.reset()` already returns an observation
  before any action. Since the observation function is deterministic,
  conditioning the uniform initial belief on it is exact Bayes: we simply
  keep the location hypotheses whose window matches.
* **Rollouts.** Beyond the tree we use a greedy-with-noise rollout policy
  (ε = 0.2 random; otherwise prefer pushes that bring a box closer to an
  uncovered goal, else move toward the nearest unfinished box, penalising
  wall bumps and deadlock-prone pushes). A purely random rollout almost never
  reaches the sparse terminal reward within the horizon, which starves POMCP
  of signal at short budgets.
* **Budget enforcement.** The deadline is computed with `time.monotonic()`
  *before* the search and checked both in the top-level simulation loop and
  at every tree level, so `plan()` returns within the budget (verified by
  `test_pomcp_budget_is_enforced`).
* **Model correctness.** With success probabilities forced to 1.0 both the
  real environment and the generative model are deterministic;
  `test_model_matches_env_deterministically` verifies they produce identical
  trajectories and observations action-for-action on all maps (including the
  heavy-box map), so the particle filter's generative model is exact.

## 4. Scenarios and Deviations from the Recommended Setup

* **Maps.** Two scenarios are evaluated, as required:

  ```
  single (1 robot)        multi (2 robots)
  WWWWWWW                 WWWWWWW
  W A   W                 W A A W
  W  B  W                 W B B W
  W     W                 W     W
  W  G  W                 W G G W
  WWWWWWW                 WWWWWWW
  ```

  **Deviation (reported):** the two-robot map uses two small boxes / two goals
  rather than the Assignment-1 heavy-box map. A heavy box requires both
  robots to push *from the same cell simultaneously* — under initial location
  uncertainty for **both** robots this makes episodes dominated by the
  coordination lottery rather than by the planning quality we are asked to
  measure. The full Assignment-1/2 map remains available via
  `--scenarios ex2` (the model, filter and planner support heavy boxes and
  are parity-tested on that map).
* **Randomised start (reported).** On every reset the robots are teleported
  to locations drawn uniformly from the free cells — i.e., the true initial
  state is a genuine sample from the uniform initial belief b₀. With the
  fixed map start the 30 runs would differ only through transition noise
  while the agent still believes it could be anywhere; sampling b₀ is the
  faithful POMDP simulation.

## 5. Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `time_budget_short` / `time_budget_long` | 1 s / 20 s | enforced wall-clock, per decision |
| `n_runs` | 30 | per (scenario, budget) pair |
| `particles_n` | 500 | no depletion-driven failures observed; reinvigoration triggers only rarely (deterministic observations) |
| `depth_max` (search/rollout horizon) | 60 | γ=0.99: γ⁶⁰ ≈ 0.547, so a 60-step horizon does *not* fully discount deep reward — still ≈3–5× the typical solve length, which matters more than the discount tail at this γ |
| `c` (UCB1 exploration) | 1.0 | recommended starting value; Q ∈ [0, 1] here so c = 1 explores adequately |
| `gamma` | 0.99 | **Deviation (reported):** the code defaults to 0.95 (matching previous assignments), but the reported experiments were run with `--gamma 0.99` for a longer effective planning horizon, appropriate given multi-agent episodes can need 15–30+ steps |
| Observation mode | egocentric (Alternative B) | Alternative A also implemented; select with `--obs north` |
| Rollout policy ε | 0.2 | greedy-with-noise heuristic (see §3) |
| `max_steps` (truncation) | 200 | truncated episodes count as 200 steps and as failures |
| Rejection-sampling cap | 100·N attempts | then map-based reinvigoration |

## 6. Environment Bug Found and Fixed

While debugging a `multi | 20s` run that made no progress for ~66 minutes
(200 steps × 20 s), we traced it to a bug in the **course-provided**
`environment/stochastic_env.py` (not ex4-specific code): two agents'
simultaneously-resolved actions could land on the same cell within one
`step()` call — e.g. one agent's move landing exactly where another agent's
box push lands the same step — and a later bookkeeping pass then silently
**deleted** the box sitting on that cell (nothing else tracks box positions
independently of the grid). One goal became permanently uncoverable and the
episode burned all 200 decisions before truncating.

We fixed `StochasticMultiAgentBoxPushEnv.step()` to plan every agent's
outcome against a frozen grid snapshot and cancel any action whose
destination collides with another agent's or a pushed box's destination
(both revert to a no-op, matching the environment's existing
"blocked → stay put" semantics). The original code is kept as an inline
comment directly above the fix. Regression tests are in
`tests/test_stochastic_env.py::TestSimultaneousCollisions` (confirmed to
fail against the pre-fix code). The identical fix was mirrored in our own
`pomdp_model.py` generative model, which had the same latent bug — required
to keep model/real-environment parity, on which POMCP's correctness
depends (`test_ex4.py::test_model_matches_env_deterministically`).

This bug pre-dates Assignment 4 and is not introduced by our code; it is
simply far more consequential here because episodes are long (up to
200 × 20 s) and starts are randomised, making the rare collision much more
likely to be hit and much more costly when it is.

## 7. Results

30 runs per cell; steps are real environment steps (compass actions) until
all goals are covered; truncated episodes (cap 200) count at the cap.

| Scenario | Budget | Mean steps | Std steps | Solve rate |
|---|---|---|---|---|
| single (1 robot) | 1 s | 6.83 | 3.60 | 100% (30/30) |
| single (1 robot) | 20 s | 6.57 | 3.00 | 100% (30/30) |
| multi (2 robots) | 1 s | 19.87 | 34.23 | 97% (29/30) |
| multi (2 robots) | 20 s | 15.27 | 34.45 | 97% (29/30) |

Run with: `python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4`
(base seed 0, `gamma=0.99`, `--particles 500`, `--c 1.0`, `--depth 60`).
The large std for `multi` is driven almost entirely by one truncated
(200-step) run per budget — see discussion below.

(Raw per-run data: `results.json`; full log: `results.log`.)

## 8. Discussion

**Effect of the decision-time budget (1 s vs 20 s).** For the single-robot
scenario the budget barely matters (mean 6.83 → 6.57 steps, std 3.60 → 3.00):
the map is small enough that even 1 second of POMCP search already finds a
near-optimal policy, so the extra 19 seconds mostly buys marginally lower
variance rather than materially fewer steps. For the two-robot scenario the
effect is more visible on the *typical* run — mean steps drop noticeably
(19.87 → 15.27) with the larger budget, since coordinating two agents under
independent location uncertainty gives POMCP a much larger joint
action/observation space (4² = 16 root actions, and joint belief particles)
to search, so more simulations per decision translate into materially better
joint pushes and less wasted maneuvering. What the extra budget does *not*
fix is the tail: one run per budget hit the 200-step truncation cap
regardless of how much time was allowed per decision — see below.

**Effect of the multi-agent scenario.** Two robots consistently need more
steps than one (mean ≈ 15–20 vs ≈ 7) and show far higher variance (std ≈ 34
vs ≈ 3). Two things compound: (1) each robot must localize its *own* hidden
position independently before the joint belief is sharp enough to coordinate
efficient pushes, roughly doubling the "figuring out where I am" overhead
before productive pushing starts; and (2) the belief and search space both
grow multiplicatively with the number of agents (joint particles, joint
actions), so the same particle/time budget covers proportionally less of the
outcome space per agent than in the single-robot case.

**The two truncated runs.** In both `multi` cells exactly one run (out of
30) hit the 200-step cap rather than solving — run 8/30 at 1 s, and run 6/30
at 20 s. Run 6/30 at 20 s is notable: it is the *exact seed* we used to
diagnose the environment collision bug in §6 (that bug is what caused the
original ~66-minute stall). After the fix, we verified offline with a
dedicated, uncontended process that this same seed *is* solvable — POMCP
solved it cleanly in 12 decisions (~4 minutes). In the actual 30-run sweep,
however, this episode ran with 4 worker processes sharing the machine's
cores (`--jobs 4`), so each worker's 20-second budget bought noticeably
fewer POMCP simulations than our dedicated single-process check got in the
same wall-clock time — weaker per-decision search quality pushed this
specific run past the 200-step cap live, even though the underlying scenario
is solvable in principle. This is a real, if secondary, effect of `--jobs`:
raising parallelism trades some per-decision search quality (and thus
occasional truncated runs) for total sweep wall-clock time. Beyond that
contention effect, with only 20% push failure and 20% move-deviation
probability compounding over many decisions, an unlucky sequence of failed
pushes or a slow-to-collapse belief can push an episode past 200 steps even
under full compute. A larger `--max-steps`, `--jobs 1`, more particles, or a
stronger rollout heuristic would likely reduce (but not necessarily
eliminate) this tail; we left the recommended defaults and `--jobs 4` in
place for a practical sweep runtime rather than re-tuning against two
observed outliers.

**Takeaway.** More compute per decision helps most when the search space is
large (multi-agent) and helps little when it is already small enough to
solve near-optimally within 1 second (single-agent). Neither budget rescues
a run from a genuinely unlucky stochastic trajectory — that risk is a
property of the domain (sparse terminal reward, no-pull box mechanics, push
failure probability) more than of the planner's time budget.
