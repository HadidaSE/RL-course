"""POMDP wrapper around the course's real stochastic environment.

The "real world" of Assignment 4 is still the course-provided
``StochasticMultiAgentBoxPushEnv``.  This wrapper adapts it to the POMDP
interface of the assignment:

* **Compass actions** — the wrapper accepts one of the four directions per
  agent.  MiniGrid rotations are deterministic and carry no information, so
  the wrapper sets each agent's facing direction directly and issues a
  single ``forward`` action; one compass action therefore equals one
  environment step, exactly matching the transition model of Assignment 2.
* **Partial observations** — instead of MiniGrid's default 7x7
  facing-forward view (which matches neither alternative in the
  assignment), the wrapper slices a 3x3 egocentric window (Alternative B)
  straight out of the full grid around each agent's *true but hidden*
  location, exactly like the board representation the PDDL extractor uses.
* **Hidden initial location** — with ``randomize_start=True`` the agents
  are teleported to locations drawn uniformly from the free cells after
  every reset, i.e. the true initial state is a sample from the uniform
  initial belief b0.

The planner must never read ``agent_positions`` from this wrapper; it only
receives the observations returned by :meth:`reset` and :meth:`step`.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv
from pomdp_model import (
    BOX,
    FREE,
    OBS_OFFSETS,
    WALL,
    Cell,
    JointAction,
    Observation,
    State,
    Window,
)


class BoxPushPOMDPEnv:
    """Real-environment adapter exposing the POMDP interface of ex4."""

    def __init__(
        self,
        ascii_map: Sequence[str],
        max_steps: int = 200,
        randomize_start: bool = True,
        seed: Optional[int] = None,
        obs_mode: str = "egocentric",
    ) -> None:
        """Creates the wrapped stochastic environment.

        Args:
            ascii_map: Map in the course ASCII format.
            max_steps: Episode truncation limit (in compass actions).
            randomize_start: If True, agent start locations are re-sampled
                uniformly from the free cells on every reset (a draw from
                the uniform initial belief).
            seed: Seed for the start-location sampler and the env's
                stochastic dynamics.
            obs_mode: Observation alternative — "egocentric" (Alternative B,
                agent-centred 3x3 window) or "north" (Alternative A, 3x3
                window immediately north of the agent).
        """
        if obs_mode not in OBS_OFFSETS:
            raise ValueError(f"unknown obs_mode {obs_mode!r}; "
                             f"expected one of {sorted(OBS_OFFSETS)}")
        self.obs_offsets = OBS_OFFSETS[obs_mode]
        self.env = StochasticMultiAgentBoxPushEnv(
            ascii_map=list(ascii_map), max_steps=max_steps
        )
        self.randomize_start = randomize_start
        self._rng = random.Random(seed)
        self._seed = seed
        self.agents = list(self.env.possible_agents)
        self.n_agents = len(self.agents)

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Observation, dict]:
        """Resets the world and returns the initial joint observation."""
        if self._seed is not None:
            np.random.seed(self._seed)
            self._seed = None  # only the first reset is pinned
        self.env.reset()
        if self.randomize_start:
            self._teleport_agents()
        return self._observation(), {}

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Observation, float, bool, bool, dict]:
        """Executes one compass action per agent in the real environment.

        Args:
            actions: Mapping ``agent name -> direction`` (0=right, 1=down,
                2=left, 3=up).

        Returns:
            ``(observation, reward, terminated, truncated, info)`` with a
            single scalar team reward (1.0 when all goals are covered).
        """
        for agent, direction in actions.items():
            self.env.agent_dirs[agent] = direction
            self.env.agent_objects[agent].dir = direction
        forward = {agent: 2 for agent in actions}
        _, rewards, terms, truncs, infos = self.env.step(forward)
        reward = 1.0 if any(r > 0 for r in rewards.values()) else 0.0
        return (
            self._observation(),
            reward,
            any(terms.values()),
            any(truncs.values()),
            infos,
        )

    # ------------------------------------------------------------------
    # Ground-truth access (debugging / evaluation only — not for planning)
    # ------------------------------------------------------------------

    def true_state(self) -> State:
        """Returns the hidden ground-truth state (for logging/tests only)."""
        small, heavy = [], []
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell = self.env.core_env.grid.get(x, y)
                if cell is not None and getattr(cell, "type", "") == "box":
                    if getattr(cell, "box_size", "") == "heavy":
                        heavy.append((x, y))
                    else:
                        small.append((x, y))
        positions = tuple(self.env.agent_positions[a] for a in self.agents)
        return State(positions, tuple(sorted(small)), tuple(sorted(heavy)))

    # ------------------------------------------------------------------
    # Observation function (Alternative B) over the real grid
    # ------------------------------------------------------------------

    def _observation(self) -> Observation:
        """Slices one 3x3 window per agent from the true grid.

        The window's placement (agent-centred / north of the agent) follows
        the configured ``obs_mode``.  Goal cells and agents read as FREE so
        the window depends only on the agent's own hidden location and the
        (known-dynamics) box layout — a deterministic observation function,
        as required.
        """
        return tuple(
            self._window(self.env.agent_positions[agent]) for agent in self.agents
        )

    def _window(self, pos: Cell) -> Window:
        grid = self.env.core_env.grid
        x0, y0 = pos
        cells = []
        for dx, dy in self.obs_offsets:
            x, y = x0 + dx, y0 + dy
            if not (0 <= x < self.env.width and 0 <= y < self.env.height):
                cells.append(WALL)
                continue
            obj = grid.get(x, y)
            if obj is None:
                cells.append(FREE)
            elif obj.type == "wall":
                cells.append(WALL)
            elif obj.type == "box":
                cells.append(BOX)
            else:  # goal, agent sprites, ...
                cells.append(FREE)
        return tuple(cells)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _teleport_agents(self) -> None:
        """Moves every agent to a uniformly drawn free cell (b0 sample)."""
        grid = self.env.core_env.grid
        free = [
            (x, y)
            for y in range(self.env.height)
            for x in range(self.env.width)
            if grid.get(x, y) is None
            or getattr(grid.get(x, y), "type", "") in ("goal", "agent")
        ]
        chosen = self._rng.sample(free, self.n_agents)
        for agent, new_pos in zip(self.agents, chosen):
            old_pos = self.env.agent_positions[agent]
            if grid.get(*old_pos) is self.env.agent_objects[agent]:
                grid.set(*old_pos, None)
            self.env.agent_positions[agent] = new_pos
            grid.set(*new_pos, self.env.agent_objects[agent])
