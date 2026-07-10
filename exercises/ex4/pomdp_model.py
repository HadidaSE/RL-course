"""Generative model of the Box Pushing POMDP (Assignment 4).

This module implements a lightweight, pure-Python simulator of the Box
Pushing domain that mirrors the dynamics of
``environment.stochastic_env.StochasticMultiAgentBoxPushEnv`` exactly, but
operates on a compact, hashable state representation.  POMCP needs to run
thousands of simulations per second, which the full MiniGrid environment
cannot sustain — hence this dedicated generative model, used both for
planning (tree simulations and rollouts) and for the particle-filter belief
update, as required by the original POMCP algorithm (Silver & Veness, 2010).

Actions are *compass moves* — the four directions the PDDL abstraction of
previous assignments used (there is no notion of a facing direction; the
rotations of the underlying MiniGrid env are deterministic bookkeeping and
are absorbed into the wrapper in :mod:`pomdp_env`).

Transition model (identical to Assignment 2):
    * move:  0.8 intended direction, 0.1 deviation 90° left, 0.1 right.
      A deviation into a blocked cell leaves the agent in place.
    * push (small box, or heavy box by two co-located agents): 0.8 success,
      0.2 no state change.
    * unmet preconditions → deterministic no-op.

Observation model (Alternative B — egocentric):
    A deterministic 3x3 window centred on the agent's true location.  Each
    cell is labelled FREE / WALL / BOX; goal cells and other agents read as
    FREE (the observation carries no information about other agents, so it
    stays deterministic given the agent's own location and the box layout).
"""

from __future__ import annotations

import random
from itertools import product
from typing import Iterable, List, NamedTuple, Sequence, Tuple

Cell = Tuple[int, int]
Window = Tuple[int, ...]
Observation = Tuple[Window, ...]
JointAction = Tuple[int, ...]

# Observation cell labels.
FREE, WALL, BOX = 0, 1, 2

# Compass directions, indexed like MiniGrid's DIR_TO_VEC:
# 0 = right, 1 = down, 2 = left, 3 = up.
DIR_VECS: Tuple[Cell, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIR_NAMES: Tuple[str, ...] = ("right", "down", "left", "up")

# The two observation alternatives of the assignment, expressed as the
# (dx, dy) offsets of the 3x3 window relative to the agent's true location,
# in row-major order (northernmost row first):
#   * "egocentric" (Alternative B) — the agent sits in the centre cell;
#   * "north" (Alternative A) — the window lies immediately to the north of
#     the agent (its bottom row is the row directly above the agent),
#     regardless of any facing direction.
OBS_OFFSETS: dict = {
    "egocentric": tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)),
    "north": tuple((dx, dy) for dy in (-3, -2, -1) for dx in (-1, 0, 1)),
}


class State(NamedTuple):
    """Compact, hashable world state.

    Attributes:
        agents: Per-agent (x, y) positions, in fixed agent order.
        small: Sorted tuple of small-box positions.
        heavy: Sorted tuple of heavy-box positions.
    """

    agents: Tuple[Cell, ...]
    small: Tuple[Cell, ...]
    heavy: Tuple[Cell, ...]


class BoxPushModel:
    """Generative model ``G(s, a) -> (s', o, r)`` for the Box Pushing POMDP.

    The map layout (walls, goals) is static and fully known; the only
    initial uncertainty is over the agents' own locations, but box positions
    are carried inside each state because push outcomes depend on the hidden
    agent locations.
    """

    def __init__(
        self,
        ascii_map: Sequence[str],
        move_success_prob: float = 0.8,
        push_success_prob: float = 0.8,
        obs_mode: str = "egocentric",
    ) -> None:
        """Parses the ASCII map and precomputes static layout information.

        Args:
            ascii_map: Rows of the map; 'W' wall, 'G' goal, 'B' small box,
                'C' heavy box, 'A' agent start, ' ' free.
            move_success_prob: Probability a move goes in the intended
                direction (the remainder is split between the two sides).
            push_success_prob: Probability a push succeeds.
            obs_mode: Observation alternative — "egocentric" (Alternative B,
                agent-centred window) or "north" (Alternative A, window
                immediately north of the agent).
        """
        if obs_mode not in OBS_OFFSETS:
            raise ValueError(f"unknown obs_mode {obs_mode!r}; "
                             f"expected one of {sorted(OBS_OFFSETS)}")
        self.ascii_map = list(ascii_map)
        self.width = len(ascii_map[0])
        self.height = len(ascii_map)
        self.move_success_prob = move_success_prob
        self.push_success_prob = push_success_prob
        self.obs_mode = obs_mode
        self.obs_offsets = OBS_OFFSETS[obs_mode]

        self.walls: frozenset = frozenset()
        self.goals: frozenset = frozenset()
        walls, goals, small, heavy, agents = set(), set(), [], [], []
        for y, row in enumerate(ascii_map):
            for x, char in enumerate(row):
                if char == "W":
                    walls.add((x, y))
                elif char == "G":
                    goals.add((x, y))
                elif char == "B":
                    small.append((x, y))
                elif char == "C":
                    heavy.append((x, y))
                elif char == "A":
                    agents.append((x, y))

        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.initial_small: Tuple[Cell, ...] = tuple(sorted(small))
        self.initial_heavy: Tuple[Cell, ...] = tuple(sorted(heavy))
        self.map_agent_starts: Tuple[Cell, ...] = tuple(agents)
        self.n_agents = len(agents)

        # All non-wall cells (an agent may also stand on a goal cell).
        self.open_cells: Tuple[Cell, ...] = tuple(
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.walls
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def initial_state(self, agent_positions: Sequence[Cell]) -> State:
        """Builds a state with the given agent positions and the known
        initial box layout."""
        return State(tuple(agent_positions), self.initial_small, self.initial_heavy)

    def agent_cells(self, small: Iterable[Cell], heavy: Iterable[Cell]) -> List[Cell]:
        """Returns cells an agent could occupy given a box configuration."""
        boxes = set(small) | set(heavy)
        return [c for c in self.open_cells if c not in boxes]

    def is_terminal(self, state: State) -> bool:
        """True iff every goal cell is covered by a box (and goals exist)."""
        boxes = set(state.small) | set(state.heavy)
        return bool(self.goals) and self.goals <= boxes

    # ------------------------------------------------------------------
    # Observation function (Alternative B "egocentric" / Alternative A
    # "north", selected by ``obs_mode``)
    # ------------------------------------------------------------------

    def window(self, pos: Cell, boxes: frozenset) -> Window:
        """Computes the deterministic 3x3 window for an agent at ``pos``.

        The window's placement relative to the agent is given by
        ``obs_mode``: centred on the agent (Alternative B) or immediately
        to the north of it (Alternative A).

        Args:
            pos: The agent's true (x, y) location.
            boxes: Set of all box positions (small and heavy).

        Returns:
            A 9-tuple in row-major order (northernmost row first); each
            entry is FREE, WALL or BOX.  Out-of-bounds cells read as WALL
            (relevant for the "north" mode near the top border).
        """
        x0, y0 = pos
        cells = []
        for dx, dy in self.obs_offsets:
            x, y = x0 + dx, y0 + dy
            if not (0 <= x < self.width and 0 <= y < self.height) or (x, y) in self.walls:
                cells.append(WALL)
            elif (x, y) in boxes:
                cells.append(BOX)
            else:
                cells.append(FREE)
        return tuple(cells)

    def observe(self, state: State) -> Observation:
        """Returns the joint observation: one egocentric window per agent."""
        boxes = frozenset(state.small) | frozenset(state.heavy)
        return tuple(self.window(pos, boxes) for pos in state.agents)

    # ------------------------------------------------------------------
    # Transition function
    # ------------------------------------------------------------------

    def step(
        self, state: State, actions: JointAction, rng: random.Random
    ) -> Tuple[State, Observation, float, bool]:
        """Samples one transition of the generative model.

        Mirrors ``StochasticMultiAgentBoxPushEnv.step`` pass-for-pass:
        heavy-box joint pushes are resolved first, then the remaining agents
        act one at a time in index order (agents never block one another —
        the underlying env clears agent sprites during movement resolution).

        Args:
            state: Current world state.
            actions: Per-agent compass direction (0=right, 1=down, 2=left,
                3=up), in agent order.
            rng: Random source used for all stochastic outcomes.

        Returns:
            Tuple ``(next_state, observation, reward, done)`` where reward
            is 1.0 exactly when all goals become covered (terminal).
        """
        agents = list(state.agents)
        small = set(state.small)
        heavy = set(state.heavy)

        # -- Pass 1: heavy-box joint pushes -----------------------------
        consumed = set()
        if heavy:
            groups: dict = {}
            for i, d in enumerate(actions):
                vec = DIR_VECS[d]
                target = (agents[i][0] + vec[0], agents[i][1] + vec[1])
                if target in heavy:
                    groups.setdefault(target, []).append(i)
            for box_pos, pushers in groups.items():
                if len(pushers) < 2:
                    continue
                origins = {agents[i] for i in pushers}
                dirs = {actions[i] for i in pushers}
                if len(origins) == 1 and len(dirs) == 1:
                    vec = DIR_VECS[next(iter(dirs))]
                    dest = (box_pos[0] + vec[0], box_pos[1] + vec[1])
                    if self.is_open(dest, small, heavy):
                        if rng.random() < self.push_success_prob:
                            heavy.remove(box_pos)
                            heavy.add(dest)
                            for i in pushers:
                                agents[i] = box_pos
                # Matching the env: once >= 2 agents target the same heavy
                # box their intents are consumed, whatever the outcome.
                consumed.update(pushers)

        # -- Pass 2: individual moves / small-box pushes -----------------
        #
        # ================================================================
        # BUG (mirrors the one found and fixed in
        # ``environment/stochastic_env.py`` on 2026-07-10): the ORIGINAL
        # loop below mutates `agents`/`small` incrementally, one agent at a
        # time, so a later agent in the loop can react to an earlier
        # agent's move — but an EARLIER agent has no way to know where a
        # LATER agent (or the box it pushes) will land in this same
        # simultaneous step. Two entities could be resolved onto the same
        # cell (e.g. one agent's own move landing exactly on a cell a
        # different agent pushes a box to), which is physically impossible
        # and, in the real environment, silently destroyed the box. Left
        # unfixed here, this model would disagree with the now-fixed real
        # environment on exactly these collision cases, breaking the
        # generative-model / real-environment parity POMCP relies on.
        #
        # ORIGINAL (buggy) implementation — kept here for reference only,
        # do not re-enable:
        #
        # for i, d in enumerate(actions):
        #     if i in consumed:
        #         continue
        #     pos = agents[i]
        #     vec = DIR_VECS[d]
        #     target = (pos[0] + vec[0], pos[1] + vec[1])
        #
        #     if target in small:
        #         behind = (target[0] + vec[0], target[1] + vec[1])
        #         if self.is_open(behind, small, heavy):
        #             if rng.random() < self.push_success_prob:
        #                 small.remove(target)
        #                 small.add(behind)
        #                 agents[i] = target
        #     elif self.is_open(target, small, heavy):
        #         actual = self._sample_move_dir(d, rng)
        #         avec = DIR_VECS[actual]
        #         atarget = (pos[0] + avec[0], pos[1] + avec[1])
        #         if self.is_open(atarget, small, heavy):
        #             agents[i] = atarget
        #     # else: wall or lone heavy-box push → no-op.
        #
        # ================================================================
        # FIX: plan every agent's outcome against a frozen snapshot (no
        # incremental mutation of `agents`/`small` while planning), collect
        # every cell each plan claims (an agent's own landing cell, plus a
        # pushed box's landing cell), and cancel any plan that claims a
        # cell also claimed by another plan — both parties simply stay put,
        # matching the existing "blocked → no-op" semantics. Identical
        # logic to the real environment's fix, so model and environment
        # agree on every collision case.
        # ================================================================

        planned = {}  # agent index -> resolved outcome
        for i, d in enumerate(actions):
            if i in consumed:
                continue
            pos = agents[i]
            vec = DIR_VECS[d]
            target = (pos[0] + vec[0], pos[1] + vec[1])

            if target in small:
                behind = (target[0] + vec[0], target[1] + vec[1])
                if self.is_open(behind, small, heavy):
                    if rng.random() < self.push_success_prob:
                        planned[i] = {
                            "kind": "push",
                            "agent_to": target,
                            "box_from": target,
                            "box_to": behind,
                        }
                    # push fails → agent stays, world unchanged (no plan)
            elif self.is_open(target, small, heavy):
                actual = self._sample_move_dir(d, rng)
                avec = DIR_VECS[actual]
                atarget = (pos[0] + avec[0], pos[1] + avec[1])
                if self.is_open(atarget, small, heavy):
                    planned[i] = {"kind": "move", "agent_to": atarget}
                # else: deviated into obstacle → agent stays (no plan)
            # else: wall or lone heavy-box push → no-op (no plan).

        claims: dict = {}  # cell -> list of agent indices whose plan claims it
        for i, plan in planned.items():
            claimed_cells = {plan["agent_to"]}
            if plan["kind"] == "push":
                claimed_cells.add(plan["box_to"])
            for cell in claimed_cells:
                claims.setdefault(cell, []).append(i)

        collided = {
            i for agents_here in claims.values() if len(agents_here) > 1
            for i in agents_here
        }

        for i, plan in planned.items():
            if i in collided:
                continue
            if plan["kind"] == "push":
                small.remove(plan["box_from"])
                small.add(plan["box_to"])
            agents[i] = plan["agent_to"]

        next_state = State(tuple(agents), tuple(sorted(small)), tuple(sorted(heavy)))
        done = self.is_terminal(next_state)
        reward = 1.0 if done else 0.0
        return next_state, self.observe(next_state), reward, done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def is_open(self, cell: Cell, small: set, heavy: set) -> bool:
        """True iff ``cell`` is inside the map, not a wall and not a box."""
        x, y = cell
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and cell not in self.walls
            and cell not in small
            and cell not in heavy
        )

    def _sample_move_dir(self, intended: int, rng: random.Random) -> int:
        """Samples the actual travel direction of a move action."""
        r = rng.random()
        side = (1.0 - self.move_success_prob) / 2.0
        if r < self.move_success_prob:
            return intended
        if r < self.move_success_prob + side:
            return (intended - 1) % 4
        return (intended + 1) % 4


def joint_actions(n_agents: int) -> List[JointAction]:
    """Enumerates all joint compass actions for ``n_agents`` agents."""
    return list(product(range(4), repeat=n_agents))
