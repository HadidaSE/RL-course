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
# baseline sweep (§7)
python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4 \
    --out exercises/ex4/sweep_baseline
# optional enhancements (§8)
python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4 --improved \
    --out exercises/ex4/sweep_improved
python3 exercises/ex4/test_ex4.py            # sanity tests (7 tests)
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
| `particles_n` | 500 | measured sufficient — rejection sampling refilled all 500 particles at every one of 178 logged belief updates, and reinvigoration never fired (see below) |
| `depth_max` (search/rollout horizon) | 60 | γ=0.99: γ⁶⁰ ≈ 0.547, so a 60-step horizon does *not* fully discount deep reward — still ≈3–5× the typical solve length, which matters more than the discount tail at this γ |
| `c` (UCB1 exploration) | 1.0 | recommended starting value; Q ∈ [0, 1] here so c = 1 explores adequately |
| `gamma` | 0.99 | **Deviation (reported):** the code defaults to 0.95 (matching previous assignments), but the reported experiments were run with `--gamma 0.99` for a longer effective planning horizon, appropriate given multi-agent episodes can need 15–30+ steps |
| Observation mode | egocentric (Alternative B) | Alternative A also implemented; select with `--obs north` |
| Rollout policy ε | 0.2 | greedy-with-noise heuristic (see §3) |
| `max_steps` (truncation) | 200 | truncated episodes count as 200 steps and as failures |
| Rejection-sampling cap | 100·N attempts | then map-based reinvigoration |

**Justification of `particles_n = 500` (measured).** We instrumented the
belief update (`--verbose --jobs 1`) over 16 episodes on both maps — 178
belief updates in total:

| Quantity | Result |
|---|---|
| Particles accepted per update | **500 / 500 at every single update** (min = max = 500) |
| Rejection-sampling acceptance rate | 19.1% mean; 1.0% at the worst individual step |
| Attempts used per update | 2,621 mean, 48,804 worst, against the 50,000 cap |
| Updates requiring reinvigoration | **0 / 178** |

So N = 500 is sufficient: the unweighted rejection-sampling update of Silver
& Veness never failed to refill the filter, and the map-based reinvigoration
fallback — although implemented and unit-tested
(`test_particle_filter_survives_depletion`) — was never needed on these maps.

The margin is thinner than the headline suggests, though. The worst observed
step consumed 48,804 of its 50,000 permitted attempts (97.6%), i.e. a 1.0%
acceptance rate. This is the expected consequence of a *deterministic*
observation function: immediately after a step whose real observation was
unlikely under the current belief, almost every sampled particle is rejected.
Halving N would not help — the cap scales with N — but a substantially harder
map, or an observation window carrying more information, could push a step
past the cap and into reinvigoration. That is precisely why the fallback
exists rather than being left as an unhandled edge case.

Reproduce: `python3 exercises/ex4/solution_ex4.py --scenarios single multi
--budgets 1 --runs 8 --jobs 1 --verbose --out exercises/ex4/pf_diagnostics`
(raw log: `pf_diagnostics.log`).

For scale, the same log records POMCP simulation counts per decision at the
1 s budget: **10,575 mean** (min 1,373, max 122,499) — confirming the budget
is spent on genuine search rather than on the belief update.

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
| single (1 robot) | 1 s | 6.83 | 3.75 | 100% (30/30) |
| single (1 robot) | 20 s | 6.63 | 3.04 | 100% (30/30) |
| multi (2 robots) | 1 s | 26.50 | 46.92 | 93% (28/30) |
| multi (2 robots) | 20 s | 15.70 | 34.45 | 97% (29/30) |

Run with: `python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4`
(base seed 0, `gamma=0.99`, `--particles 500`, `--c 1.0`, `--depth 60`).

Because truncated runs enter the mean at the 200-step cap, a single
truncation shifts a 30-run cell's mean by ≈6 steps and inflates its std
enormously. The `multi` std values above are therefore almost entirely a
truncation artefact, not a spread over typical episodes. The **solved-only**
statistics describe the typical run:

| Scenario | Budget | Mean steps (solved runs only) | Std | n |
|---|---|---|---|---|
| single | 1 s | 6.83 | 3.75 | 30 |
| single | 20 s | 6.63 | 3.04 | 30 |
| multi | 1 s | 14.11 | 7.42 | 28 |
| multi | 20 s | 9.34 | 4.01 | 29 |

**Reproducibility note.** An earlier execution of this identical command
(2026-07-10) gave 6.83 / 6.57 / 19.87 / 15.27. The three stable cells
reproduce to within ±0.1 steps; `multi | 1 s` moved from 19.87 to 26.50
purely because that run truncated once and this one truncated twice. Even
with fixed seeds the results are not bit-reproducible: the number of POMCP
simulations that fit inside a wall-clock budget depends on machine load, so
seeds fix the *environment* noise but not the *search* effort. Cells whose
mean is dominated by rare truncations are consequently the least stable, a
point the enhancement comparison in §8 addresses directly.

(Raw per-run data: `sweep_baseline.json`; full log: `sweep_baseline.log`.
The 2026-07-10 execution is retained as `results.json` / `results.log`.)

## 8. Optional POMCP Enhancements

Three optional improvements were added on top of the baseline POMCP. They are
**off by default** so the §7 numbers stay reproducible; `--improved` enables
all three. All are assignment-compliant: no external POMDP/planning/RL
library, no change to the reward/termination/`gamma` definition, and they
never read the agent's true hidden location — they operate only on states
sampled from the belief and on the known map.

| Flag | What it changes |
|---|---|
| `--reuse-tree` | The search tree is retained between decisions and pruned to the `T(hao)` subtree after each real action/observation, instead of being rebuilt from scratch every step (canonical Silver & Veness 2010 POMCP). |
| `--smart-rollout` | The heuristic rollout greedily matches each agent to a **distinct** unfinished box, so two robots stop chasing the same one. |
| `--preferred-actions` | On node expansion the rollout policy's greedy joint action is tried first among untried actions, warm-starting the 16-arm joint search. |

Measured with the same seeds, machine and `--jobs 4` as §7, run back-to-back
with the baseline sweep:

| Scenario | Budget | Baseline mean / std / solve | Improved mean / std / solve |
|---|---|---|---|
| single | 1 s | 6.83 / 3.75 / 100% | 6.70 / 3.29 / 100% |
| single | 20 s | 6.63 / 3.04 / 100% | 6.63 / 3.19 / 100% |
| multi | 1 s | 26.50 / 46.92 / 93% | **13.83 / 8.20 / 100%** |
| multi | 20 s | 15.70 / 34.45 / 97% | 15.63 / 34.37 / 97% |

**What actually improved.** The headline `multi | 1 s` drop (26.50 → 13.83)
is a *reliability* gain, not a speedup. Restricted to solved runs the two
planners are indistinguishable — 14.11 (n=28) vs 13.83 (n=30) at 1 s, and
9.34 vs 9.28 (both n=29) at 20 s. What changed is the tail: the baseline
truncated twice at 1 s with a worst run of 200 steps, the improved planner
truncated **zero** times with a worst run of 46. Removing two 200-step
entries is what halves the mean and collapses the std from 46.92 to 8.20.

This is the expected shape of the effect. All three enhancements make each
second of search go further — reused statistics, less duplicated effort
between the two robots, a sensible first action at every new node — which
matters only when compute is the binding constraint. At 1 s on the joint
(16-arm) two-robot problem it is, so the runs that would otherwise flounder
into the cap now finish. At 20 s the baseline search is already good enough
that the extra efficiency buys nothing measurable (15.70 → 15.63, both with
one truncation), and on the single-robot map neither budget was ever
compute-starved, so both planners sit at ≈6.7 steps.

Raw data: `sweep_improved.json` / `sweep_improved.log`. Reproduce with:

```bash
python3 exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4 --improved --out exercises/ex4/sweep_improved
```

## 9. Discussion

**Effect of the decision-time budget (1 s vs 20 s).** For the single-robot
scenario the budget barely matters (6.83 → 6.63 steps): the map is small
enough that even 1 second of POMCP search already finds a near-optimal
policy, so the extra 19 seconds change nothing measurable.

For the two-robot scenario the budget matters a great deal, but the raw
means hide it — 26.50 → 15.70 looks like a large gain and is partly a
truncation artefact. The **solved-only** means isolate the real effect:
**14.11 → 9.34 steps**, a ~34% reduction on typical episodes, with std
falling 7.42 → 4.01. This is the genuine budget effect, and it is much
cleaner than the raw table suggests. It is also exactly where more search
*should* pay: coordinating two agents under independent location uncertainty
gives POMCP a 4² = 16-arm joint action space over joint belief particles, so
additional simulations per decision translate directly into better joint
pushes and less wasted maneuvering.

The budget also reduces, but does not eliminate, the truncation tail: 2/30
truncations at 1 s versus 1/30 at 20 s.

**Effect of the multi-agent scenario.** Two robots consistently need more
steps than one — ≈9–14 versus ≈6.7 on solved runs — and are far more prone
to the occasional catastrophic episode (all three truncations across the
whole sweep are `multi` runs; `single` never truncated in 60 episodes). Two
things compound: (1) each robot must localize its *own* hidden position
before the joint belief is sharp enough to coordinate efficient pushes,
roughly doubling the "figuring out where I am" overhead before productive
pushing starts; and (2) the belief and search space both grow
multiplicatively with the number of agents (joint particles, joint actions),
so the same particle and time budget covers proportionally less of the
outcome space per agent than in the single-robot case.

**The truncated runs.** Three episodes out of 120 hit the 200-step cap,
all in `multi`: two at 1 s and one at 20 s. Two mechanisms produce them.
First, contention: the sweep runs 4 worker processes (`--jobs 4`), so each
worker's wall-clock budget buys noticeably fewer POMCP simulations than a
dedicated process would. We confirmed this concretely — the seed that
truncated in an earlier 20 s sweep was re-run in a single uncontended
process and solved cleanly in 12 decisions. Raising `--jobs` therefore
trades per-decision search quality for total sweep runtime. Second, plain
bad luck: with 20% push failure and 20% move deviation compounding over many
decisions, an unlucky sequence or a slow-to-collapse belief can exhaust 200
steps even under full compute.

Notably, the enhancements of §8 address the *first* mechanism and not the
second — they eliminated both 1 s truncations (where search effort was the
binding constraint) while leaving the 20 s truncation in place (where it was
not). A larger `--max-steps`, `--jobs 1`, or more particles would further
shrink this tail without removing it.

**Takeaway.** More compute per decision helps most when the search space is
large (multi-agent, 14.11 → 9.34 steps) and essentially not at all when it
is already small enough to solve near-optimally within 1 second
(single-agent, 6.83 → 6.63). The same principle explains §8: making each
second of search go further reproduces the benefit of a larger budget
exactly where compute was scarce, and nowhere else. What no time budget
rescues is a genuinely unlucky stochastic trajectory — that risk is a
property of the domain (sparse terminal reward, no-pull box mechanics, push
failure probability) rather than of the planner.
