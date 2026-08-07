"""POMCP — Partially Observable Monte-Carlo Planning (Silver & Veness, 2010).

Implemented from scratch (no POMDP/planning libraries), operating directly
over the particle representation of the belief:

* every planning step samples a start state from the current particle
  filter and runs one simulation through the search tree;
* inside the tree, actions are chosen with the UCB1 rule
  ``Q(ha) + c * sqrt(log N(h) / N(ha))``;
* the first time a simulation leaves the tree a single new node is added
  and the remaining value is estimated by a depth-limited **rollout** with
  a simple heuristic policy;
* returns are backed up through all visited nodes;
* the wall-clock decision budget is enforced inside the search loop —
  simulations stop the moment the deadline passes.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from pomdp_model import (
    DIR_VECS,
    BoxPushModel,
    Cell,
    JointAction,
    Observation,
    State,
    joint_actions,
)

RolloutPolicy = Callable[[State, random.Random], JointAction]


class _Node:
    """History (belief) node of the POMCP search tree.

    Attributes:
        visits: N(h) — number of simulations through this history.
        children: One ``_ActionEdge`` per joint action, or None while the
            node has not yet been expanded.
    """

    __slots__ = ("visits", "children", "preferred")

    def __init__(self) -> None:
        self.visits = 0
        self.children: Optional[List["_ActionEdge"]] = None
        self.preferred: Optional[frozenset] = None


class _ActionEdge:
    """Action edge ha with its running statistics and observation children."""

    __slots__ = ("action", "visits", "value", "children")

    def __init__(self, action: JointAction) -> None:
        self.action = action
        self.visits = 0
        self.value = 0.0
        self.children: Dict[Observation, _Node] = {}


class POMCPPlanner:
    """Online POMCP planner for the Box Pushing POMDP."""

    def __init__(
        self,
        model: BoxPushModel,
        gamma: float = 0.95,
        exploration_c: float = 1.0,
        max_depth: int = 60,
        rollout_policy: Optional[RolloutPolicy] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Configures the planner.

        The search tree is retained across decisions and pruned to the
        ``T(hao)`` subtree after each real action/observation (canonical
        POMCP, Silver & Veness 2010), so prior simulations are reused; call
        :meth:`advance` after each real step and :meth:`reset` at the start of
        each episode.  When a node is expanded the rollout policy's greedy
        joint action is marked "preferred" and tried first among untried
        actions, warm-starting the search.

        Args:
            model: Generative model ``G(s, a) -> (s', o, r)``.
            gamma: Discount factor (as in previous assignments).
            exploration_c: UCB1 exploration constant ``c``.
            max_depth: Search/rollout horizon; simulations are cut off once
                ``depth >= max_depth`` (where gamma**depth is negligible).
            rollout_policy: Policy used beyond the tree; defaults to the
                greedy-with-noise :class:`HeuristicRolloutPolicy`.
            rng: Random source (a fresh one is created if omitted).
        """
        self.model = model
        self.gamma = gamma
        self.exploration_c = exploration_c
        self.max_depth = max_depth
        self.rng = rng or random.Random()
        self.actions = joint_actions(model.n_agents)
        self.rollout_policy = rollout_policy or HeuristicRolloutPolicy(model)
        self._root: Optional[_Node] = None
        self.last_stats: dict = {}

    def reset(self) -> None:
        """Discards any retained tree; call at the start of each episode."""
        self._root = None

    def advance(self, action: JointAction, observation: Observation) -> None:
        """Prunes the retained tree to the ``T(hao)`` subtree.

        If the real observation was never simulated (so no matching child
        exists), the tree is dropped and the next :meth:`plan` starts fresh.

        Args:
            action: The joint action actually executed.
            observation: The real observation returned by the environment.
        """
        if self._root is None or self._root.children is None:
            self._root = None
            return
        edge = next((e for e in self._root.children if e.action == action), None)
        self._root = edge.children.get(observation) if edge else None

    def plan(self, particles: Sequence[State], time_budget: float) -> JointAction:
        """Selects an action for the current belief within a time budget.

        Continues from the retained ``T(hao)`` subtree (or a fresh root if
        none) and runs simulations until the wall-clock deadline expires.

        Args:
            particles: Current belief — the particle filter's particles.
            time_budget: Decision budget in seconds, enforced inside the
                simulation loop.

        Returns:
            The joint action with the highest estimated value at the root.
        """
        deadline = time.monotonic() + time_budget
        root = self._root if self._root is not None else _Node()
        n_simulations = 0
        while time.monotonic() < deadline:
            state = self.rng.choice(particles)
            self._simulate(state, root, 0, deadline)
            n_simulations += 1

        self._root = root
        best = self._best_root_action(root)
        self.last_stats = {
            "simulations": n_simulations,
            "root_visits": root.visits,
            "root_values": {
                edge.action: (round(edge.value, 4), edge.visits)
                for edge in (root.children or [])
            },
        }
        return best

    def _simulate(
        self, state: State, node: _Node, depth: int, deadline: float
    ) -> float:
        """Runs one simulation from ``state`` at ``node``; returns the
        discounted return observed from this depth downward."""
        if depth >= self.max_depth or time.monotonic() >= deadline:
            return 0.0

        if node.children is None:
            node.children = [_ActionEdge(a) for a in self.actions]
            node.visits = 1
            node.preferred = self._preferred(state)
            return self._rollout(state, depth)

        edge = self._ucb_select(node)
        next_state, obs, reward, done = self.model.step(state, edge.action, self.rng)

        if done:
            value = reward
        else:
            child = edge.children.get(obs)
            if child is None:
                child = _Node()
                edge.children[obs] = child
            value = reward + self.gamma * self._simulate(
                next_state, child, depth + 1, deadline
            )

        node.visits += 1
        edge.visits += 1
        edge.value += (value - edge.value) / edge.visits
        return value

    def _preferred(self, state: State) -> Optional[frozenset]:
        """The rollout policy's greedy joint action(s) at ``state``, if any."""
        greedy = getattr(self.rollout_policy, "greedy", None)
        if greedy is None:
            return None
        return frozenset({greedy(state)})

    def _ucb_select(self, node: _Node) -> _ActionEdge:
        """UCB1 action selection; untried actions are tried first, and among
        untried actions the "preferred" ones (if any) are tried first."""
        untried = [e for e in node.children if e.visits == 0]
        if untried:
            if node.preferred:
                pref = [e for e in untried if e.action in node.preferred]
                if pref:
                    return self.rng.choice(pref)
            return self.rng.choice(untried)
        log_n = math.log(node.visits)
        return max(
            node.children,
            key=lambda e: e.value
            + self.exploration_c * math.sqrt(log_n / e.visits),
        )

    def _rollout(self, state: State, depth: int) -> float:
        """Depth-limited rollout with the heuristic policy."""
        total = 0.0
        discount = 1.0
        while depth < self.max_depth:
            action = self.rollout_policy(state, self.rng)
            state, _, reward, done = self.model.step(state, action, self.rng)
            total += discount * reward
            if done:
                break
            discount *= self.gamma
            depth += 1
        return total

    def _best_root_action(self, root: _Node) -> JointAction:
        """Highest-value visited root action (random if nothing was tried,
        which can only happen with a degenerate time budget)."""
        visited = [e for e in (root.children or []) if e.visits > 0]
        if not visited:
            return self.rng.choice(self.actions)
        best_value = max(e.value for e in visited)
        best = [e for e in visited if e.value == best_value]
        return self.rng.choice(best).action


class HeuristicRolloutPolicy:
    """Greedy-with-noise rollout policy.

    With probability ``epsilon`` a uniformly random joint action is taken;
    otherwise each agent picks the direction with the best myopic score:

    * pushing a box so that it gets closer to an uncovered goal is best;
    * otherwise, moving closer (Manhattan) to its assigned box is preferred;
    * bumping into walls and non-improving pushes are penalised.

    With more than one agent, agents are first greedily matched to *distinct*
    unfinished boxes so two robots do not chase the same box.

    Rollouts legitimately use the *sampled* state — POMCP rollouts always
    run on fully specified states drawn from the belief.
    """

    PUSH_GOOD = 100.0
    PUSH_BAD = -50.0
    BLOCKED = -1000.0

    def __init__(self, model: BoxPushModel, epsilon: float = 0.2) -> None:
        """Creates the rollout policy.

        Args:
            model: Generative model (for map/goal geometry).
            epsilon: Probability of a uniformly random joint action.
        """
        self.model = model
        self.epsilon = epsilon
        self.all_actions = joint_actions(model.n_agents)

    def __call__(self, state: State, rng: random.Random) -> JointAction:
        if rng.random() < self.epsilon:
            return rng.choice(self.all_actions)
        return self.greedy(state, rng)

    def greedy(
        self, state: State, rng: Optional[random.Random] = None
    ) -> JointAction:
        """Deterministic (greedy) joint action; ties broken by ``rng`` if
        given, else by the first candidate (so it is usable as a stable
        "preferred action" outside a rollout)."""
        assignment = self._assign(state) if self.model.n_agents > 1 else None
        return tuple(
            self._best_direction(
                state, i, rng, assignment.get(i) if assignment else None
            )
            for i in range(self.model.n_agents)
        )

    def _assign(self, state: State) -> dict:
        """Greedily matches each agent to a distinct unfinished box."""
        positions = state.agents
        pending = [
            b for b in (set(state.small) | set(state.heavy))
            if b not in self.model.goals
        ]
        assignment: dict = {}
        used: set = set()
        pairs = sorted(
            (self._man(positions[i], b), i, b)
            for i in range(len(positions))
            for b in pending
        )
        for _, i, b in pairs:
            if i in assignment or b in used:
                continue
            assignment[i] = b
            used.add(b)
        for i in range(len(positions)):
            if i not in assignment and pending:
                assignment[i] = min(pending, key=lambda b: self._man(positions[i], b))
        return assignment

    def _best_direction(
        self,
        state: State,
        agent_idx: int,
        rng: Optional[random.Random],
        assigned_box: Optional[Cell] = None,
    ) -> int:
        pos = state.agents[agent_idx]
        small = set(state.small)
        heavy = set(state.heavy)
        boxes = small | heavy
        open_goals = self.model.goals - boxes
        if (
            assigned_box is not None
            and assigned_box in boxes
            and assigned_box not in self.model.goals
        ):
            move_targets = [assigned_box]
        else:
            move_targets = [b for b in boxes if b not in self.model.goals]

        scores = []
        for d, vec in enumerate(DIR_VECS):
            target = (pos[0] + vec[0], pos[1] + vec[1])
            if target in small:
                behind = (target[0] + vec[0], target[1] + vec[1])
                if self.model.is_open(behind, small, heavy):
                    if open_goals and self._dist(behind, open_goals) < self._dist(
                        target, open_goals
                    ):
                        scores.append(self.PUSH_GOOD)
                    else:
                        scores.append(self.PUSH_BAD)
                else:
                    scores.append(self.BLOCKED)
            elif self.model.is_open(target, small, heavy):
                if move_targets:
                    scores.append(-float(self._dist(target, move_targets)))
                else:
                    scores.append(0.0)
            else:
                scores.append(self.BLOCKED)

        best = max(scores)
        candidates = [d for d, s in enumerate(scores) if s == best]
        return rng.choice(candidates) if rng is not None else candidates[0]

    @staticmethod
    def _man(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _dist(cell: Cell, targets) -> int:
        return min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in targets)
