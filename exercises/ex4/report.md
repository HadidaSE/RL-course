# Assignment 4 Report — POMDP Planning with POMCP in the Box Pushing Environment

Students: Itamar Hadida, Omer Aviram
---

## 1. Results Table

30 runs per cell; steps are real environment steps (compass actions) until all
goals are covered. **Mean and standard deviation are computed over solved runs
only** — a truncated episode has no meaningful "steps to solve", and at the
200-step cap a single one would shift a 30-run mean by ≈6 steps and inflate the
standard deviation several-fold. The solve-rate column reports how many runs
that excludes.

| Scenario | Time limit | Mean steps | Std steps | Solve rate |
|---|---|---|---|---|
| single (1 robot) | 1 s | 6.70 | 3.29 | 100% (30/30) |
| single (1 robot) | 20 s | 6.63 | 3.19 | 100% (30/30) |
| multi (2 robots) | 1 s | 13.83 | 8.20 | 100% (30/30) |
| multi (2 robots) | 20 s | 9.28 | 3.08 | 97% (29/30) |

(`budgets = [1, 20]` `runs = 30` `jobs = 4` `base seed = 0`, `gamma=0.99`, `--particles 500`, `--c 1.0`, `--depth 60`).

For reference, the same sweep with truncated runs folded in at the 200-step cap
gives `multi | 20 s` a mean of 15.63 and a standard deviation of 34.37 — both
driven entirely by that one unsolved episode. This is precisely why the
headline table excludes them. The harness records both views in the results
JSON (`mean_steps` / `std_steps` over solved runs, `mean_steps_all` /
`std_steps_all` over all runs), alongside the raw per-run step counts.

## 2. Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `time_budget_short` / `time_budget_long` | 1 s / 20 s | enforced wall-clock, per decision |
| `n_runs` | 30 | per (scenario, budget) pair |
| `particles_n` | 500 | measured sufficient — see below |
| `depth_max` (search/rollout horizon) | 60 | γ=0.99: γ⁶⁰ ≈ 0.547, so a 60-step horizon does *not* fully discount deep reward — still ≈3–5× the typical solve length, which matters more than the discount tail at this γ |
| `c` (UCB1 exploration) | 1.0 | recommended starting value; Q ∈ [0, 1] here so c = 1 explores adequately |
| `gamma` | 0.99 | **Deviation (reported):** previous assignments use 0.95; we use 0.99 for a longer effective planning horizon, appropriate given multi-agent episodes can need 15–30+ steps |
| Observation | Alternative B (egocentric 3×3) | justification in §4 |
| Rollout policy ε | 0.2 | greedy-with-noise heuristic (§1.3) |
| `max_steps` (truncation) | 200 | truncated episodes count as failures and are excluded from the mean/std (§2) |
| Rejection-sampling cap | 100·N attempts | then map-based reinvigoration |


## 3. Choice of Observation Alternative — B (egocentric)

The assignment requires **one** of the two alternatives. We implement
**Alternative B (egocentric)**: a 3×3 window centred on the agent's true but
hidden location. Alternative A would place the same window immediately north of
the agent, independent of any facing direction.

Alternative B was chosen for three reasons:

1. **Alignment with the action space.** The actions are four direct compass
   moves with no facing direction, so all four adjacent cells are equally
   relevant to a decision. The egocentric window observes exactly those four
   cells; a north-facing window reports on the cell the agent may enter only
   when it moves north, and says nothing about the other three.
2. **No degenerate observations at the border.** A window offset three rows
   north falls entirely outside a wall-bordered board whenever the agent is in
   the top playable row, yielding an all-WALL reading. That reading is identical
   for every cell in that row, so the observation is uninformative precisely
   where the belief most needs constraining. An egocentric window is always
   in-bounds and therefore never degenerates in this way.
3. **Faster belief contraction.** Because each observation constrains the
   agent's immediate neighbourhood on all four sides, fewer candidate cells
   remain consistent with it. Under a deterministic observation function this
   directly reduces the number of location hypotheses surviving each
   particle-filter update, so the belief localises in fewer steps.

## 4. Discussion of the Results

**Effect of the decision-time limit (1 s vs 20 s).** For the single-robot
scenario the budget barely matters (6.70 → 6.63 steps): the map is small enough
that even 1 second of POMCP search already finds a near-optimal policy, so the
extra 19 seconds change nothing measurable. The single-agent problem is simply
never compute-starved.

For the two-robot scenario the budget matters a great deal: **13.83 → 9.28
steps**, a ~33% reduction, with the standard deviation falling 8.20 → 3.08.
This is exactly where more search *should* pay: coordinating two agents under
independent location uncertainty gives POMCP a 4² = 16-arm joint action space
over joint belief particles, so additional simulations per decision translate
directly into better joint pushes and less wasted maneuvering. The worst solved
run also shrinks sharply — 46 steps at 1 s versus 14 steps at 20 s — so the
larger budget buys consistency as much as speed. In short, the time limit
improves decision *quality* only where the search space is large enough for the
planner to be compute-bound.


---