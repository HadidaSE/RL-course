# How to Run Assignment 4 (POMCP)

All commands below are run from the **repository root** using the project's
virtualenv (`.venv`). If your venv lives elsewhere, replace the interpreter
path accordingly.

## 1. Sanity tests (fast, ~1 minute)

Verifies the generative model exactly matches the real environment, the
observation function, particle-filter tracking/depletion handling, the POMCP
time budget, and that a full episode solves end-to-end:

```bash
.venv/bin/python exercises/ex4/test_ex4.py
```

Expected output: six `... OK` lines and `All tests passed.`

## 2. Quick smoke test (~15 seconds)

Runs 3 episodes per scenario with a tiny 0.1 s decision budget, just to see
the whole pipeline working:

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --quick
```

## 3. The full assignment experiment

30 runs per (scenario, budget) cell, budgets of 1 s and 20 s per decision,
both the single-robot and the two-robot scenarios:

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4
```

* `--jobs 4` runs 4 episodes in parallel (one per performance core on this
  machine) — keep `jobs` ≤ physical cores so every decision still gets a full
  core and the time budget stays meaningful. Use `--jobs 1` for the most
  faithful (but slowest) timing.
* Estimated wall-clock time with `--jobs 4`: the 1 s cells take a few minutes
  each; the 20 s cells take roughly 20–40 minutes each (≈ 1–1.5 h total).
  With `--jobs 1` expect ~4–5 hours total.

If you prefer to run the cells separately (e.g., the long ones overnight):

```bash
# fast cells first
.venv/bin/python exercises/ex4/solution_ex4.py --budgets 1  --runs 30 --jobs 4 --out exercises/ex4/results_s1
# long cells later
.venv/bin/python exercises/ex4/solution_ex4.py --budgets 20 --runs 30 --jobs 4 --out exercises/ex4/results_s20
```

### Outputs

For output prefix `results` (the default, written into `exercises/ex4/`):

| File | Contents |
|------|----------|
| `results.log` | Full log — per-run progress lines and the final summary table (mean steps, std, solve rate). |
| `results.json` | Raw machine-readable results, including per-run step counts, useful for plots. |

After the run, copy the numbers from the `RESULTS SUMMARY` table into the
results table in `report.md` (section 6) and write the discussion
(section 7).

## 4. Useful options

```text
--scenarios single multi ex2   which maps to run (ex2 = the Assignment-1/2
                               heavy-box map; much harder, off by default)
--budgets 1 20                 per-decision wall-clock budgets (seconds)
--runs 30                      episodes per (scenario, budget) cell
--particles 500                particle filter size N
--c 1.0                        UCB1 exploration constant
--depth 60                     POMCP search / rollout horizon
--gamma 0.95                   discount factor
--max-steps 200                episode truncation limit
--obs egocentric               observation function: 'egocentric' =
                               Alternative B (3x3 window centred on the
                               agent, the default used for the report) or
                               'north' = Alternative A (3x3 window
                               immediately north of the agent)
--seed 0                       base seed (run i uses seed base+10000*i)
--jobs 1                       parallel episodes
--reuse-tree                   retain the POMCP tree across decisions and
                               prune it to T(hao) each step (canonical POMCP)
--smart-rollout                task-allocation rollout: match each agent to a
                               distinct unfinished box
--preferred-actions            try the greedy joint action first when a tree
                               node is expanded
--improved                     enable all three enhancements at once
                               (--reuse-tree --smart-rollout --preferred-actions)
--quick                        3 runs at 0.1 s budget (smoke test)
--out PREFIX                   output prefix for the .log/.json files
```

Example — watch a single verbose episode's belief behaviour: run the tests
or a `--quick` run and read the DEBUG lines in the `.log` file (particle
acceptance/reinvigoration counts and POMCP simulation counts per step).

Example — compare the two observation alternatives head-to-head:

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --quick --obs egocentric --out exercises/ex4/smoke_ego
.venv/bin/python exercises/ex4/solution_ex4.py --quick --obs north      --out exercises/ex4/smoke_north
```

(In our smoke runs the north window localizes noticeably more slowly — it
carries no information about the cells east/west/south of the agent — which
is part of why Alternative B is the default.)

## 5. What to check while it runs

* `solved=True` on (nearly) all runs — a `solved=False` line means the
  episode hit the 200-step cap and is counted at 200 steps.

Per-step particle-filter diagnostics are **not** emitted by default. They
require `--verbose`, and because pool workers do not inherit the parent's
logging configuration they only reach the log file under `--jobs 1`:

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --scenarios single multi \
    --budgets 1 --runs 8 --jobs 1 --verbose --out exercises/ex4/pf_diagnostics
```

Then check the DEBUG lines for `pf_accepted` close to N and `reinvig=0` —
persistent reinvigoration, or `attempts` approaching the 100·N cap, would
suggest raising `--particles`. Our measured baseline (178 belief updates):
accepted 500/500 every time, reinvigoration 0, acceptance rate 19.1% mean
but only 1.0% at the worst step. See `report.md` §5.

## 6. Optional POMCP enhancements (`pomcp.py`, `solution_ex4.py`)

Three optional planner improvements were added on top of the baseline POMCP.
They are **off by default** so the original results stay reproducible, and each
has its own flag (or use `--improved` to turn on all three). All three are
assignment-compliant: they do not touch the reward/termination/`gamma`
definition, use no external POMDP/planning/RL library, and never read the
agent's true hidden location — they operate only on states sampled from the
belief and on the known map/box layout.

| Flag | What changed | Where |
|------|--------------|-------|
| `--reuse-tree` | The search tree is retained between decisions instead of rebuilt from scratch every step. After the real action `a` and observation `o`, the tree is pruned to the `T(hao)` subtree so earlier simulations are reused (canonical Silver & Veness 2010 POMCP). | `POMCPPlanner.plan` (reuses `self._root`), new `POMCPPlanner.reset()` / `POMCPPlanner.advance(action, obs)`; the online loop in `run_episode` calls `planner.reset()` at episode start and `planner.advance(...)` after each belief update. |
| `--smart-rollout` | The heuristic rollout now greedily matches each agent to a **distinct** unfinished box (nearest-first), so two robots stop chasing the same box. Matching is hand-rolled (no `scipy`). | `HeuristicRolloutPolicy` gains `assign_tasks`, `_assign()` and a deterministic `greedy()`; `_best_direction()` takes an optional `assigned_box`. |
| `--preferred-actions` | When a tree node is expanded, the rollout policy's greedy joint action is marked "preferred" and tried first among the untried actions, warm-starting the 16-arm joint search. | `_Node.preferred`, `POMCPPlanner._preferred()`, and the preferred-first branch in `POMCPPlanner._ucb_select`. |

Config plumbing lives in `solution_ex4.py`: the `ExperimentConfig` fields
`reuse_tree` / `smart_rollout` / `preferred_actions`, the matching CLI flags
(plus `--improved`), and their wiring into the `POMCPPlanner` and rollout
policy inside `run_episode`.

Run the improved planner over the full experiment (writing to a separate
prefix so the baseline files are untouched):

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --budgets 1 20 --runs 30 --jobs 4 --improved --out exercises/ex4/sweep_improved
```

Measured effect over the full 30-run sweep (baseline and improved run
back-to-back on the same machine with the same seeds and `--jobs 4`):

| Scenario | Budget | Baseline | Improved |
|---|---|---|---|
| single | 1 s | 6.83 / std 3.75 / 100% | 6.70 / std 3.29 / 100% |
| single | 20 s | 6.63 / std 3.04 / 100% | 6.63 / std 3.19 / 100% |
| multi | 1 s | 26.50 / std 46.92 / 93% | **13.83 / std 8.20 / 100%** |
| multi | 20 s | 15.70 / std 34.45 / 97% | 15.63 / std 34.37 / 97% |

The large `multi 1s` gain is a **reliability** effect, not a speedup: on
solved runs alone the two planners are indistinguishable (14.11 vs 13.83
steps). The baseline truncated twice at 1 s (worst run 200 steps), the
improved planner zero times (worst run 46) — dropping those two 200-step
entries is what halves the mean. See `report.md` §8.
