import numpy as np
from minigrid.core.constants import DIR_TO_VEC
from environment.multi_agent_env import MultiAgentBoxPushEnv


class StochasticMultiAgentBoxPushEnv(MultiAgentBoxPushEnv):
    """
    Stochastic version of MultiAgentBoxPushEnv — used in Assignment 2.

    Transition model
    ----------------
    move  (agent moves into an empty/goal cell):
        0.8  → agent moves in the intended direction
        0.1  → agent moves 90° to the left  of the intended direction
        0.1  → agent moves 90° to the right of the intended direction
        If the deviated cell is blocked (wall / box / other agent) the agent
        stays put without error.

    push-small / push-heavy  (precondition fully met):
        0.8  → push succeeds  (same outcome as the deterministic env)
        0.2  → push fails, no state change

    If the precondition of any action is NOT met the action has no effect
    (deterministic no-op), exactly as in the deterministic environment.
    """

    def __init__(
        self,
        ascii_map=None,
        max_steps=100,
        render_mode=None,
        move_success_prob=0.8,
        push_success_prob=0.8,
    ):
        super().__init__(
            ascii_map=ascii_map, max_steps=max_steps, render_mode=render_mode
        )
        self.move_success_prob = move_success_prob
        self.push_success_prob = push_success_prob

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_move_dir(self, intended_dir):
        """
        Sample the actual travel direction for a *move* action.

        Returns
        -------
        int  — one of the four MiniGrid directions (0=right,1=down,2=left,3=up)
        """
        side_prob = (1.0 - self.move_success_prob) / 2.0
        r = np.random.random()
        if r < self.move_success_prob:
            return intended_dir
        elif r < self.move_success_prob + side_prob:
            return (intended_dir - 1) % 4   # 90° left of intended
        else:
            return (intended_dir + 1) % 4   # 90° right of intended

    def _apply_goal_termination(self, rewards, terminations):
        """Mark all agents as terminated with reward 1."""
        for a in self.agents:
            rewards[a] = 1.0
            terminations[a] = True

    # ------------------------------------------------------------------
    # Core step override
    # ------------------------------------------------------------------

    def step(self, actions):  # noqa: C901  (complexity is inherent here)
        self.steps += 1

        observations = {}
        rewards      = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations  = {agent: False for agent in self.agents}
        infos        = {agent: {} for agent in self.agents}

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # ── Clear agent sprites from the grid ─────────────────────────
        for agent in self.agents:
            px, py = self.agent_positions[agent]
            if self.core_env.grid.get(px, py) is self.agent_objects[agent]:
                self.core_env.grid.set(px, py, None)

        # ── Pass 1: Gather forward intents ────────────────────────────
        agent_intents = {}
        for agent, action in actions.items():
            pos = self.agent_positions[agent]
            d   = self.agent_dirs[agent]

            if action == 0:    # rotate left
                self.agent_dirs[agent] = (d - 1) % 4
                self.agent_objects[agent].dir = self.agent_dirs[agent]
            elif action == 1:  # rotate right
                self.agent_dirs[agent] = (d + 1) % 4
                self.agent_objects[agent].dir = self.agent_dirs[agent]
            elif action == 2:  # forward — record intent
                vec     = DIR_TO_VEC[d]
                fwd_pos = (pos[0] + vec[0], pos[1] + vec[1])
                agent_intents[agent] = {"target_pos": fwd_pos, "dir": d, "vec": vec}

        # ── Pass 2: Heavy-box push resolution (stochastic 0.8 / 0.2) ─
        heavy_box_pushes = {}
        for agent, intent in agent_intents.items():
            fwd_pos  = intent["target_pos"]
            fwd_cell = self.core_env.grid.get(*fwd_pos)
            if fwd_cell is not None and getattr(fwd_cell, "box_size", "") == "heavy":
                heavy_box_pushes.setdefault(fwd_pos, []).append((agent, intent))

        for box_pos, pushers in heavy_box_pushes.items():
            if len(pushers) >= 2:
                origins = {self.agent_positions[a] for a, _ in pushers}
                dirs    = {i["dir"] for _, i in pushers}

                # Precondition: same origin cell, same direction
                if len(dirs) == 1 and len(origins) == 1:
                    push_dir = next(iter(dirs))
                    vec      = DIR_TO_VEC[push_dir]
                    nx, ny   = box_pos[0] + vec[0], box_pos[1] + vec[1]
                    n_cell   = self.core_env.grid.get(nx, ny)

                    # Precondition: destination must be free
                    if n_cell is None or n_cell.can_overlap():
                        # ── Stochastic push ──────────────────────────
                        if np.random.random() < self.push_success_prob:
                            box_obj = self.core_env.grid.get(*box_pos)
                            self.core_env.grid.set(*box_pos, None)
                            self.core_env.grid.set(nx, ny, box_obj)
                            for agent, _ in pushers:
                                self.agent_positions[agent] = box_pos
                        # push fails → agents stay, world unchanged

                # Either way, these agents' intents are consumed
                for agent, _ in pushers:
                    agent_intents.pop(agent, None)

        # ── Pass 3: Individual forward actions (stochastic) ───────────
        #
        # ================================================================
        # BUG (found while debugging Assignment 4 / POMCP, 2026-07-10):
        # the ORIGINAL code below resolves each agent's move/push ONE AT A
        # TIME, checking a destination cell only against static grid
        # content (walls/boxes already on the grid). It never checks
        # whether a *different* agent's simultaneous move, or a box a
        # different agent pushes in this SAME step, would land on the
        # exact same cell. When that collision happens, both entities end
        # up "at" the same (x, y): a box object sits there on the grid, and
        # a colliding agent's position in `self.agent_positions` ALSO
        # points there. The "Generate observations" block further down
        # then does, for every agent:
        #     self.core_env.grid.set(*pos, None)
        # which blindly clears whatever occupies that agent's cell before
        # placing the agent sprite — silently deleting the box, since
        # nothing else in this codebase tracks box positions independently
        # of the grid. One goal becomes permanently uncoverable and the
        # episode runs to max_steps without ever raising an error.
        #
        # ORIGINAL (buggy) implementation — kept here for reference only,
        # do not re-enable:
        #
        # for agent, intent in agent_intents.items():
        #     pos          = self.agent_positions[agent]
        #     intended_dir = intent["dir"]
        #     fwd_pos      = intent["target_pos"]
        #     vec          = intent["vec"]
        #     fwd_cell     = self.core_env.grid.get(*fwd_pos)
        #
        #     if fwd_cell is not None and getattr(fwd_cell, "box_size", "") == "small":
        #         # ── PUSH-SMALL ────────────────────────────────────────
        #         fwd_fwd_pos  = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
        #         fwd_fwd_cell = self.core_env.grid.get(*fwd_fwd_pos)
        #
        #         # Precondition: cell behind box must be free
        #         if fwd_fwd_cell is None or fwd_fwd_cell.can_overlap():
        #             # ── Stochastic push ──────────────────────────────
        #             if np.random.random() < self.push_success_prob:
        #                 self.core_env.grid.set(*fwd_fwd_pos, fwd_cell)
        #                 self.core_env.grid.set(*fwd_pos, None)
        #                 self.agent_positions[agent] = fwd_pos
        #             # push fails → agent stays, world unchanged
        #
        #     elif fwd_cell is None or fwd_cell.can_overlap():
        #         # ── MOVE ─────────────────────────────────────────────
        #         # Precondition met — apply directional stochasticity
        #         actual_dir     = self._sample_move_dir(intended_dir)
        #         actual_vec     = DIR_TO_VEC[actual_dir]
        #         actual_fwd_pos = (pos[0] + actual_vec[0], pos[1] + actual_vec[1])
        #         actual_fwd_cell = self.core_env.grid.get(*actual_fwd_pos)
        #
        #         if actual_fwd_cell is None or actual_fwd_cell.can_overlap():
        #             self.agent_positions[agent] = actual_fwd_pos
        #         # else: deviated into obstacle → agent stays, no error
        #
        #     # else: intended cell is a wall / other obstacle → no-op
        #
        # ================================================================
        # FIX: resolve every agent's move/push against a *frozen* snapshot
        # of the grid (no incremental mutation while planning — matches
        # the "simultaneous joint action" semantics this whole environment
        # already assumes elsewhere, e.g. the heavy-box joint push). For
        # each agent we compute a *plan* without touching the grid yet:
        #   - "push": box moves box_from → box_to, agent moves to box_from
        #   - "move": agent moves to agent_to
        # We then collect every cell a plan would claim (an agent's own
        # landing cell, plus a pushed box's landing cell) and cancel —
        # revert to a no-op, exactly like the existing "blocked → stay put,
        # no error" behaviour — every plan that claims a cell also claimed
        # by another plan. This also naturally covers two agents pushing
        # the *same* box in the same step (both would claim that box's
        # cell as their own landing cell) and two boxes being pushed onto
        # the same destination cell. Only collision-free plans are applied.
        # ================================================================

        planned = {}  # agent -> resolved outcome (kind, agent_to, [box_from, box_to, box_obj])
        for agent, intent in agent_intents.items():
            pos          = self.agent_positions[agent]
            intended_dir = intent["dir"]
            fwd_pos      = intent["target_pos"]
            vec          = intent["vec"]
            fwd_cell     = self.core_env.grid.get(*fwd_pos)

            if fwd_cell is not None and getattr(fwd_cell, "box_size", "") == "small":
                # ── PUSH-SMALL ────────────────────────────────────────
                fwd_fwd_pos  = (fwd_pos[0] + vec[0], fwd_pos[1] + vec[1])
                fwd_fwd_cell = self.core_env.grid.get(*fwd_fwd_pos)

                # Precondition: cell behind box must be free
                if fwd_fwd_cell is None or fwd_fwd_cell.can_overlap():
                    # ── Stochastic push ──────────────────────────────
                    if np.random.random() < self.push_success_prob:
                        planned[agent] = {
                            "kind": "push",
                            "agent_to": fwd_pos,
                            "box_from": fwd_pos,
                            "box_to": fwd_fwd_pos,
                            "box_obj": fwd_cell,
                        }
                    # push fails → agent stays, world unchanged (no plan)

            elif fwd_cell is None or fwd_cell.can_overlap():
                # ── MOVE ─────────────────────────────────────────────
                # Precondition met — apply directional stochasticity
                actual_dir     = self._sample_move_dir(intended_dir)
                actual_vec     = DIR_TO_VEC[actual_dir]
                actual_fwd_pos = (pos[0] + actual_vec[0], pos[1] + actual_vec[1])
                actual_fwd_cell = self.core_env.grid.get(*actual_fwd_pos)

                if actual_fwd_cell is None or actual_fwd_cell.can_overlap():
                    planned[agent] = {"kind": "move", "agent_to": actual_fwd_pos}
                # else: deviated into obstacle → agent stays, no error (no plan)

            # else: intended cell is a wall / other obstacle → no-op (no plan)

        # Collect every cell each plan claims (its own landing cell, plus a
        # pushed box's landing cell).
        claims = {}  # cell -> list of agents whose plan claims it
        for agent, plan in planned.items():
            claimed_cells = {plan["agent_to"]}
            if plan["kind"] == "push":
                claimed_cells.add(plan["box_to"])
            for cell in claimed_cells:
                claims.setdefault(cell, []).append(agent)

        collided_agents = {
            agent
            for agents_here in claims.values()
            if len(agents_here) > 1
            for agent in agents_here
        }

        # Apply only collision-free plans; colliding agents simply stay put.
        for agent, plan in planned.items():
            if agent in collided_agents:
                continue
            if plan["kind"] == "push":
                self.core_env.grid.set(*plan["box_from"], None)
                self.core_env.grid.set(*plan["box_to"], plan["box_obj"])
            self.agent_positions[agent] = plan["agent_to"]

        # ── Check if all boxes are on goal positions → terminate ──────
        if self._all_boxes_on_goals():
            self._apply_goal_termination(rewards, terminations)

        # ── Truncation check ──────────────────────────────────────────
        if self.steps >= self.max_steps:
            for a in self.agents:
                truncations[a] = True

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        # ── Generate observations ─────────────────────────────────────
        for agent in self.possible_agents:
            if agent in actions:
                pos = self.agent_positions[agent]
                self.core_env.agent_pos = pos
                self.core_env.agent_dir = self.agent_dirs[agent]
                self.core_env.grid.set(*pos, None)
                observations[agent] = self.core_env.gen_obs()
                self.core_env.grid.set(*pos, self.agent_objects[agent])

        return observations, rewards, terminations, truncations, infos
