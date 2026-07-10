"""Unweighted particle filter for the Box Pushing POMDP belief state.

Implements the belief representation and update of the original POMCP
algorithm (Silver & Veness, 2010):

* the belief is a bag of ``N`` unweighted particles, each a full world
  state (hypothesised agent locations + box layout);
* initialisation spreads the agent-location hypotheses uniformly over all
  free cells of the board;
* after a real step ``(a, o)`` the belief is updated by **rejection
  sampling** through the same generative model POMCP plans with: sample a
  particle, simulate ``a``, keep the successor iff its simulated
  observation equals the real one.

Because the observation function is deterministic the rejection rate can be
high and the filter can deplete entirely.  Depletion is handled by
**particle reinvigoration** that exploits full knowledge of the board map:
we enumerate the states that are exactly consistent with the received
observation (per agent, every free cell whose 3x3 window matches) and fill
the filter from that set.
"""

from __future__ import annotations

import random
from itertools import product
from typing import List, Optional

from pomdp_model import BoxPushModel, JointAction, Observation, State


class ParticleFilter:
    """Bag-of-particles belief over Box Pushing world states."""

    def __init__(
        self,
        model: BoxPushModel,
        n_particles: int = 500,
        max_attempts_factor: int = 100,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Creates an (empty) particle filter.

        Args:
            model: Generative model shared with the POMCP planner.
            n_particles: Number of particles ``N`` maintained after every
                update.
            max_attempts_factor: Rejection sampling is capped at
                ``max_attempts_factor * N`` simulations before falling back
                to map-based reinvigoration.
            rng: Random source (a fresh one is created if omitted).
        """
        self.model = model
        self.n_particles = n_particles
        self.max_attempts_factor = max_attempts_factor
        self.rng = rng or random.Random()
        self.particles: List[State] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Spreads location hypotheses uniformly over all free cells.

        Box positions are fully known initially, so every particle carries
        the map's initial box layout; only the agent locations vary.  For
        multiple agents each particle draws distinct cells per agent
        (matching how the real environment places its agents).
        """
        cells = self.model.agent_cells(
            self.model.initial_small, self.model.initial_heavy
        )
        self.particles = [
            self.model.initial_state(self.rng.sample(cells, self.model.n_agents))
            for _ in range(self.n_particles)
        ]

    def condition_on_observation(self, real_obs: Observation) -> None:
        """Filters the current particles against an observation (no action).

        Used once, right after ``env.reset()``: the initial observation is
        received before any action is taken, and because the observation
        function is deterministic, exact Bayes conditioning of the uniform
        initial belief reduces to keeping only the consistent particles.
        The filter is refilled to ``N`` from the consistent set.

        Args:
            real_obs: The joint observation returned by ``env.reset()``.
        """
        boxes = frozenset(self.model.initial_small) | frozenset(
            self.model.initial_heavy
        )
        cells = self.model.agent_cells(
            self.model.initial_small, self.model.initial_heavy
        )
        per_agent = [
            [c for c in cells if self.model.window(c, boxes) == real_obs[i]]
            for i in range(self.model.n_agents)
        ]
        consistent = [
            self.model.initial_state(combo) for combo in product(*per_agent)
        ]
        if consistent:
            self.particles = [
                self.rng.choice(consistent) for _ in range(self.n_particles)
            ]

    # ------------------------------------------------------------------
    # Belief update
    # ------------------------------------------------------------------

    def update(self, action: JointAction, real_obs: Observation) -> dict:
        """Updates the belief after a real step, by rejection sampling.

        Args:
            action: The joint compass action actually executed.
            real_obs: The joint observation returned by the real
                environment.

        Returns:
            Small stats dict (``accepted``, ``attempts``, ``reinvigorated``)
            useful for logging and for tuning ``n_particles``.
        """
        if not self.particles:
            raise RuntimeError("Particle filter used before initialize().")

        new_particles: List[State] = []
        attempts = 0
        max_attempts = self.max_attempts_factor * self.n_particles
        while len(new_particles) < self.n_particles and attempts < max_attempts:
            state = self.rng.choice(self.particles)
            next_state, obs, _, _ = self.model.step(state, action, self.rng)
            if obs == real_obs:
                new_particles.append(next_state)
            attempts += 1

        stats = {
            "accepted": len(new_particles),
            "attempts": attempts,
            "reinvigorated": 0,
        }

        if len(new_particles) < self.n_particles:
            added = self._reinvigorate(new_particles, action, real_obs)
            stats["reinvigorated"] = added

        self.particles = new_particles
        return stats

    # ------------------------------------------------------------------
    # Reinvigoration
    # ------------------------------------------------------------------

    def _reinvigorate(
        self,
        new_particles: List[State],
        action: JointAction,
        real_obs: Observation,
    ) -> int:
        """Refills a (partially) depleted filter from consistent states.

        Leverages full knowledge of the board map: for each plausible box
        configuration, every free cell whose egocentric window matches the
        agent's real observation is an exactly-consistent hypothesis.  Box
        configurations are taken from the surviving particles when any
        exist, otherwise from one-step simulations of the previous belief.

        Args:
            new_particles: Partially filled particle list (mutated in
                place, up to ``n_particles`` entries).
            action: The executed joint action.
            real_obs: The real joint observation to match.

        Returns:
            Number of particles added by reinvigoration.
        """
        if new_particles:
            configs = {(s.small, s.heavy) for s in new_particles}
        else:
            samples = [
                self.model.step(self.rng.choice(self.particles), action, self.rng)[0]
                for _ in range(min(len(self.particles), 200))
            ]
            configs = {(s.small, s.heavy) for s in samples}

        consistent: List[State] = []
        for small, heavy in configs:
            boxes = frozenset(small) | frozenset(heavy)
            cells = self.model.agent_cells(small, heavy)
            per_agent = [
                [c for c in cells if self.model.window(c, boxes) == real_obs[i]]
                for i in range(self.model.n_agents)
            ]
            # Agents may legally share a cell (joint heavy-box pushes), so
            # co-located combinations are kept.
            for combo in product(*per_agent):
                consistent.append(State(tuple(combo), small, heavy))

        target = self.n_particles
        added = 0
        if consistent:
            while len(new_particles) < target:
                new_particles.append(self.rng.choice(consistent))
                added += 1
        elif new_particles:
            # No exactly-consistent state found (should not happen with a
            # correct model) — pad by duplicating survivors.
            while len(new_particles) < target:
                new_particles.append(self.rng.choice(new_particles))
                added += 1
        else:
            # Last resort: restart from the uniform belief.
            cells = self.model.agent_cells(
                self.model.initial_small, self.model.initial_heavy
            )
            while len(new_particles) < target:
                positions = self.rng.sample(cells, self.model.n_agents)
                new_particles.append(self.model.initial_state(positions))
                added += 1
        return added
