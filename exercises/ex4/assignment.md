# Reinforcement Learning — Programming Assignment 4

## Planning Under Partial Observability of Agent Location — POMDP and POMCP in the Box Pushing Environment

---

## Background and Purpose

In previous assignments, you solved the Box Pushing problem under the assumption of **full observability**: at every moment you knew the complete state of the world with certainty, including your own location. In this assignment, you will add a new layer of uncertainty to the problem: the agent (the robot) does **not** know its initial location on the board, and must estimate its location while acting, based solely on the partial observations it collects along the way.

Such a problem is defined as a **Partially Observable Markov Decision Process (POMDP)**. You are required to solve it using **POMCP — Partially Observable Monte-Carlo Planning**, an online planning algorithm based on Monte Carlo Tree Search (MCTS) that operates directly over a **particle representation** of the belief state, without requiring an explicit, full probabilistic representation over all states.

---

## POMDP Definition

### State Space

The state space is essentially identical to the state space you are familiar with from previous assignments, except now the agent's own location is unknown. That is, for each agent (robot) separately, the space of possible locations is the set of all free cells on the board — the number of possible states equals the number of free cells on the board.

The locations of the boxes and other world elements remain fully known and fixed; the uncertainty is limited solely to the agent's own location.

### Actions and the Transition Function

The definition of actions is identical to the definition in Assignment 1. The probabilistic dynamics are identical to those defined in Assignment 2:

- **Move action:** succeeds with probability 0.8 in the intended direction; with probability 0.1 a deviation occurs to one side, and with probability 0.1 a deviation occurs to the other side.
- **Push action:** succeeds with probability 0.8; with probability 0.2 the action fails with no change of state.
- If the action's preconditions are not met (e.g., moving into a wall) — the action does not change the world state.

### Observation Function

At each step, the agent receives only a partial observation — a 3x3 "window" of the board — instead of knowing its exact location. You must implement **one** of the following two alternatives, choose one, and justify your choice in the report:

**Alternative A — Fixed directional observation:**
The observation is always a 3x3 window located to the north of the agent's actual location, regardless of the direction the agent is "facing."

**Alternative B — Egocentric observation (centered on the agent):**
The observation is a 3x3 window in which the agent is located in the center cell.

In both cases, the content of the observation (which cells are free, which are walls, and whether there is a box) is deterministic with respect to the agent's true location: O(o | s', a) = 1 if and only if o is exactly the appropriate window around s'; otherwise O(o | s', a) = 0. In other words, there is no sensor noise — the only source of uncertainty is the lack of knowledge of the initial location, together with the probabilistic dynamics of the actions themselves.

> **Important note:** MiniGrid's "default" (built-in) observation is **not** identical to either of the two alternatives described above. MiniGrid's default is a window that always faces forward relative to the direction the agent is "looking" (facing-forward, direction-dependent, 7x7 by default) — not a fixed-compass-direction window, and not a symmetric window centered on the agent. Since in our Box Pushing environment the move actions are direct (left/right/up/down, with no notion of agent "facing direction"), neither of the two alternatives can be obtained "for free" via a built-in MiniGrid wrapper such as `ViewSizeWrapper`. You must implement the observation function as your own dedicated function that performs a direct slice from the full board representation already known to the system (e.g., as already exposed through the function that generates the PDDL files) around the robot's true — but hidden from the agent — location. Verify against your actual environment implementation which MiniGrid components, if any, are relevant for you.

### Reward, Termination Conditions, and Discount Factor

Identical to the definition in previous assignments (including gamma).

---

## What You Must Implement

### 1. Observation Function
According to one of the two alternatives detailed above.

### 2. Particle Filter for the Belief State

- **Representation:** a collection of N particles, where each particle represents a possible hypothesis for the agent's location.
- **Initialization:** at the start of a run, when the agent does not know its location, the particles should be spread uniformly over all free cells on the board.
- **Belief state update** (after performing a real action *a* and receiving a real observation *o* from the environment): according to the original POMCP algorithm (Silver & Veness, 2010), this is an **unweighted rejection-sampling update**: sample a particle from the current particle filter, simulate action *a* on it using the same generative model used by POMCP for planning (transition + observation), and check whether the simulated observation obtained equals the real observation *o*. If so — keep the resulting state as a new particle; if not — reject it and sample again. The process repeats until N new particles have been collected.
- **Note:** since the observation function you defined is deterministic, the rejection rate may be high, and in edge cases the particle filter may become **completely depleted** (if no particle produced a matching observation). You must handle this case in your implementation — for example, by limiting the number of sampling attempts and completing the set via **particle reinvigoration**, or by leveraging full knowledge of the board map to directly check which states remain consistent with the received observation.
- **Result of the process:** an updated belief state (state belief) — the same particle filter serves both as the true belief-state representation between environment steps, and as the internal belief-state representation at every node of the POMCP search tree.

### 3. The POMCP Algorithm

- Implement the algorithm to run directly over the particle filter (not over an explicit probability distribution), following the original approach of Silver & Veness (2010): at each planning step, sample an initial state *s* from the current particle filter, and start a simulation from it.
- As long as the simulation remains within the existing search tree (an action-observation history you have already visited) — select the next action according to the **UCT1/UCB rule**, which balances exploitation (actions with high estimated value) and exploration (actions with few visits).
- The moment the simulation exits the existing tree (a new history that has not been visited) — add a new node to the tree, and run a **rollout** (typically using a random or simple heuristic policy) up to a predefined search depth (horizon), in order to estimate the value from that point.
- The obtained value is propagated back up the tree, updating the statistics (visit count, estimated value) of all nodes the simulation passed through.
- After the allotted decision time budget is exhausted, the action with the highest estimated value at the root of the tree is actually selected. After executing it and receiving a real observation, the belief state must be updated as described in the previous section, and the process moves to the next planning step.
- **Do not use ready-made POMDP / Planning / RL libraries** (such as `py_pomdp`) — you must implement POMCP, the particle filter, and the observation function yourselves.

---

## Online Planning Loop — General Structure

The general loop replaces the classic planning loop you are familiar with from previous assignments (where a PDDL plan was executed at each step) with a POMCP-based planning loop, operating over a belief state instead of over a known state:

```python
obs, info = env.reset()
particles = init_uniform_particles(free_cells)
done = False
while not done:
    # 1. Plan: run POMCP over the current belief state
    action = run_pomcp(particles, time_budget=time_limit)
    # 2. Execute the real action in the environment, get a real observation
    obs, reward, terminated, truncated, info = env.step(action)
    # 3. Update the belief state given the action and the observation received
    particles = update_particles(particles, action, obs)
    done = terminated or truncated
```

---

## Running the Experiment

You must run the algorithm on the problem under **two different computation time limits**, per single decision: **1 second (s1)** and **20 seconds (s20)**. Your implementation must ensure that the decision is indeed reached within the specified time limit — that is, the budget must be actually enforced inside the POMCP search tree, and not merely stated as a comment.

For each of the two time limits, run the algorithm in the simulator **30 times**, and report the average number of steps and the standard deviation required to solve the task.

---

## Recommended Starting Parameters

| Parameter | Recommended Value / Note |
|---|---|
| `time_budget_short` | 1 second |
| `time_budget_long` | 20 seconds |
| `n_runs` | 30 |
| `particles_n` | To be chosen and justified; recommended to start around 500, and increase if particle depletion is observed |
| `depth_max` / horizon | Rollout search depth (parameter required for a correct POMCP implementation); should be set according to board size and gamma |
| `c` (UCB1 exploration constant) | To be tuned; recommended to start at c ≈ 1 |
| `gamma` | As defined in previous assignments |

---

## Work Environment Setup

The work environment is identical to the one used in previous assignments (Pushing Box), and now implements a POMDP version of the same problem. If you have not yet installed the environment, see the full installation instructions in Assignment 1 or in the README.md file in the repository:

`https://github.com/ronberg-bgu/RL-course`

### Creating a Branch for This Assignment

```
git checkout -b student-{firstname}-ex4
```
Example: `student-yossi-ex4`, `student-sarah-ex4`

---

## Submission

When finished, push the branch and open a Pull Request in the course repository.

## What to Submit

1. **Full code** of: the observation function, the particle filter, and the POMCP algorithm.
2. **Results table:** average number of steps and standard deviation, for each of the two time limits (1 second, 20 seconds), for both the single-agent scenario and the two-robot scenario.
3. **Details of the hyperparameters used** (number of particles, UCB1 exploration parameter, rollout depth/length, and any additional parameter).
4. A **short explanation** of your choice of observation alternative (A or B) and the reasoning behind it.
5. A **short discussion of the results**: How did the computation time limit (1 second vs. 20 seconds) affect the quality of decisions and the average number of steps? How did the multi-agent scenario (two robots) affect performance, compared to a single agent?

---

## Important Notes

- **Do not use ready-made POMDP / Planning / RL libraries** (such as `py_pomdp` as an off-the-shelf solution) — you must implement POMCP, the particle filter, and the observation function yourselves.
- You must ensure that the computation time limit (1 second / 20 seconds) is actually enforced in the code (budget for the POMCP search tree), and not merely noted as a comment.
- The uncertainty in this assignment concerns **only** the agent's own location; the locations of the boxes and other world elements are fully known.
- Any deviation from the recommended parameters must be clearly reported, to the extent that one was made.