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
--seed 0                       base seed (run i uses seed base+10000*i)
--jobs 1                       parallel episodes
--quick                        3 runs at 0.1 s budget (smoke test)
--out PREFIX                   output prefix for the .log/.json files
```

The observation function is Alternative B (a 3x3 window centred on the agent);
see `report.md` §4 for why.

Example — watch a single verbose episode's belief behaviour: run the tests
or a `--quick` run and read the DEBUG lines in the `.log` file (particle
acceptance/reinvigoration counts and POMCP simulation counts per step).

## 5. What to check while it runs

* `solved=True` on (nearly) all runs — a `solved=False` line means the
  episode hit the 200-step cap. Such runs count as failures in the solve rate
  and are excluded from the reported mean/std (the JSON also keeps
  `mean_steps_all` / `std_steps_all`, which include them at the cap).

Per-step particle-filter diagnostics are **not** emitted by default. They
require `--verbose`, and because pool workers do not inherit the parent's
logging configuration they only reach the log file under `--jobs 1`:

```bash
.venv/bin/python exercises/ex4/solution_ex4.py --scenarios single multi \
    --budgets 1 --runs 8 --jobs 1 --verbose --out exercises/ex4/pf_diagnostics
```

Then check the DEBUG lines for `pf_accepted` close to N and `reinvig=0` —
persistent reinvigoration, or `attempts` approaching the 100·N cap, would
suggest raising `--particles`. Our measurements (178 belief updates):
accepted 500/500 every time, reinvigoration 0, acceptance rate 19.1% mean
but only 1.0% at the worst step. See `report.md` §3.

## 6. POMCP implementation details (`pomcp.py`)

Beyond the textbook POMCP loop, the planner has three built-in features (all
always on). They are assignment-compliant: they do not touch the
reward/termination/`gamma` definition, use no external POMDP/planning/RL
library, and never read the agent's true hidden location — they operate only
on states sampled from the belief and on the known map/box layout.

| Feature | What it does | Where |
|---------|--------------|-------|
| Tree reuse | The search tree is retained between decisions instead of rebuilt from scratch every step. After the real action `a` and observation `o`, the tree is pruned to the `T(hao)` subtree so earlier simulations are reused (canonical Silver & Veness 2010 POMCP). | `POMCPPlanner.plan` (reuses `self._root`), `POMCPPlanner.reset()` / `POMCPPlanner.advance(action, obs)`; the online loop in `run_episode` calls `planner.reset()` at episode start and `planner.advance(...)` after each belief update. |
| Task-allocation rollout | The heuristic rollout greedily matches each agent to a **distinct** unfinished box (nearest-first), so two robots do not chase the same box. Matching is hand-rolled (no `scipy`). | `HeuristicRolloutPolicy._assign()`, the deterministic `greedy()`, and the optional `assigned_box` argument of `_best_direction()`. |
| Preferred actions | When a tree node is expanded, the rollout policy's greedy joint action is marked "preferred" and tried first among the untried actions, warm-starting the 16-arm joint search. | `_Node.preferred`, `POMCPPlanner._preferred()`, and the preferred-first branch in `POMCPPlanner._ucb_select`. |

Measured over a full 30-run sweep (`--jobs 4`), mean/std over solved runs
only (discussion in `report.md` §2 and §5):

| Scenario | Budget | Mean steps | Std | Solve rate |
|---|---|---|---|---|
| single | 1 s | 6.70 | 3.29 | 100% |
| single | 20 s | 6.63 | 3.19 | 100% |
| multi | 1 s | 13.83 | 8.20 | 100% |
| multi | 20 s | 9.28 | 3.08 | 97% |
