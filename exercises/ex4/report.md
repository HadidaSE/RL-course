# Assignment 4 Report — POMDP Planning with POMCP in the Box Pushing Environment

Students: Itamar Hadida, Omer Aviram
---

## 1. Results Table

30 runs per cell; steps are real environment steps (compass actions) until all
goals are covered. Two of the 120 runs never solved the task and were
truncated at the 200-step cap — one in each two-robot cell (run 28 of
`multi | 1 s` and run 30 of `multi | 20 s`; different seeds, so it is not a
single systematically hard instance). The table reports **both** views: the
headline statistics over solved runs only, and the same statistics over all 30
runs with the two truncated episodes folded in at the cap.

| Scenario | Time limit | Mean steps (solved) | Std (solved) | Mean steps (all 30) | Std (all 30) | Solved |
|---|---|---|---|---|---|---|
| single (1 robot) | 1 s | 7.07 | 4.41 | 7.07 | 4.41 | 30/30 (100%) |
| single (1 robot) | 20 s | 6.67 | 3.24 | 6.67 | 3.24 | 30/30 (100%) |
| multi (2 robots) | 1 s | 13.03 | 6.40 | 19.27 | 34.15 | 29/30 (97%) |
| multi (2 robots) | 20 s | 9.59 | 3.89 | 15.93 | 34.39 | 29/30 (97%) |

(`budgets = [1, 20]` `runs = 30` `jobs = 4` `base seed = 0`, `gamma=0.99`, `--particles 500`, `--c 1.0`, `--depth 60`; standard deviations are population values, `np.std`.)

We read the **solved-only** columns as the answer to "how many steps does the
task take". A truncated episode has no meaningful "steps to solve", and at the
200-step cap a single one shifts the 30-run mean by ≈6 steps and inflates the
standard deviation almost nine-fold (6.40 → 34.15 and 3.89 → 34.39): the
all-runs columns describe the truncation cap far more than they describe the
planner. Both views come straight from the results JSON (`mean_steps` /
`std_steps` over solved runs, `mean_steps_all` / `std_steps_all` over all
runs), which also keeps the raw per-run step counts.

The two failures are not planner timeouts but **irreversible dead-ends in the
domain**: a box pushed into the border ring can never be recovered, because
freeing it would require the robot to stand inside a wall. Of the 20 free
cells on these boards only 9 are positions from which a box can still reach a
goal, so one 0.1-probability move deviation during a push — or a push made
under a wrong location hypothesis — can lose an episode outright. This is also
why the failure rate is identical (1/30) at both budgets: no amount of extra
search undoes a dead-end once it happens. Finally, step counts are not
bit-reproducible across executions — how many POMCP simulations fit inside a
wall-clock budget depends on machine load — so the table describes one
complete sweep rather than a fixed reference value.

## 2. Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `time_budget_short` / `time_budget_long` | 1 s / 20 s | enforced wall-clock, per decision |
| `n_runs` | 30 | per (scenario, budget) pair |
| `particles_n` | 500 | measured over 178 belief updates: all 500 filled by genuine rejection sampling every time (0 reinvigorations), worst-step acceptance 1.0% |
| `depth_max` (search/rollout horizon) | 60 | γ=0.99: γ⁶⁰ ≈ 0.547, so a 60-step horizon does *not* fully discount deep reward — still ≈5–9× the observed mean solve length (6.7–13.0 steps), which matters more than the discount tail at this γ |
| `c` (UCB1 exploration) | 1.0 | recommended starting value; Q ∈ [0, 1] here so c = 1 explores adequately |
| `gamma` | 0.99 | **Deviation (reported):** previous assignments use 0.95; we use 0.99 for a longer effective planning horizon, appropriate given multi-agent episodes reach 20–27 steps in our runs |
| Observation | Alternative B (egocentric 3×3) | justification in §3 |
| Rollout policy ε | 0.2 | greedy-with-noise heuristic: pushes a box toward the nearest open goal, else moves toward the robot's assigned box; with 2 robots each is assigned a distinct box |
| `max_steps` (truncation) | 200 | truncated episodes count as failures; reported both excluded from and folded into the mean/std (§1) |
| Rejection-sampling cap | 100·N attempts | our choice (assignment only requires *some* cap); sized to the measured 1.0% worst-case acceptance above, then map-based reinvigoration |
| Tree reuse | on | search tree is pruned to `T(hao)` after each real step rather than rebuilt from scratch, so earlier simulations carry over |
| Preferred actions | on | on expanding a new tree node, the rollout policy's greedy action is tried first among the untried actions |
| Start location | resampled per episode | **Deviation (reported):** each reset teleports the robot(s) to a uniformly random free cell (a true draw from the initial belief), rather than the map's fixed start cell |


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

Both budgets were run on the same 30 seeds per scenario, so the two columns
can be compared run by run, not only in aggregate.

**Effect of the decision-time limit (1 s vs 20 s).** For the single-robot
scenario the budget barely matters: 7.07 → 6.67 steps, i.e. 0.4 steps or ~6%.
The paired view makes it even clearer — of the 30 seeds, **24 produce the
identical step count** at both budgets, 4 improve and 2 get worse. The map is
small enough that 1 second of POMCP search already finds a near-optimal
policy, so the extra 19 seconds mostly trim the tail (the worst run drops from
24 to 14 steps, which is what pulls the standard deviation down from 4.41 to
3.24). The single-agent problem is simply never compute-starved.

For the two-robot scenario the budget matters a great deal: **13.03 → 9.59
steps**, a ~26% reduction, with the standard deviation falling 6.40 → 3.89.
Here the paired comparison is decisive rather than marginal: over the 28 seeds
solved at both budgets, 20 s wins on **21**, loses on 3 and ties on 4, for a
mean improvement of 3.29 steps per run. This is exactly where more search
*should* pay: coordinating two agents under independent location uncertainty
gives POMCP a 4² = 16-arm joint action space over joint belief particles, so
additional simulations per decision translate directly into better joint
pushes and less wasted maneuvering. The worst solved run shrinks too (27 → 20
steps), so the larger budget buys consistency as well as speed. In short, the
time limit improves decision *quality* only where the search space is large
enough for the planner to be compute-bound.

**Effect of the multi-agent scenario.** Two robots cost roughly **1.8× the
single-robot step count at 1 s** (13.03 vs 7.07) and **1.44× at 20 s** (9.59
vs 6.67) — and the two-robot map genuinely requires more work, since it has
two boxes to deliver instead of one. The interesting part is that the penalty
*shrinks* as the budget grows: the gap is a compute effect, not an inherent
coordination cost. With two agents the belief is a joint distribution over
both hidden locations and the action space grows from 4 to 16 arms, so at 1 s
the planner cannot cover the joint tree well and produces the kind of wasted
maneuvering that shows up as a wide spread (std 6.40, worst run 27 steps). At
20 s it covers it well enough that the two robots behave close to two
efficient single agents. The two-robot scenario is also the only one that ever
fails (29/30 at both budgets), for the dead-end reason given in §1: twice as
many boxes means twice as many chances to push one irrecoverably into the
border ring.


---