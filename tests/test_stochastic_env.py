"""
Tests for StochasticMultiAgentBoxPushEnv.

Strategy
--------
We use deterministic seeds / forced probabilities to test stochastic behaviour:
  - Set move_success_prob=1.0  → always moves intended (deterministic)
  - Set push_success_prob=1.0  → always pushes (deterministic)
  - Set push_success_prob=0.0  → push always fails
  - Set move_success_prob=0.0  → always deviates (never goes intended way)
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.stochastic_env import StochasticMultiAgentBoxPushEnv


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
SIMPLE_MAP = [
    "WWWWW",
    "W A W",
    "W B W",
    "W   W",
    "WWWWW",
]


def make_env(move_p=1.0, push_p=1.0, ascii_map=None):
    m = ascii_map if ascii_map is not None else SIMPLE_MAP
    return StochasticMultiAgentBoxPushEnv(
        ascii_map=m,
        max_steps=50,
        move_success_prob=move_p,
        push_success_prob=push_p,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestStochasticEnvAPI(unittest.TestCase):
    """Basic PettingZoo API checks."""

    def test_reset_returns_obs_for_all_agents(self):
        env = make_env()
        obs, info = env.reset()
        self.assertIn("agent_0", obs)
        self.assertIn("agent_0", info)

    def test_action_space_is_discrete_3(self):
        from gymnasium import spaces

        env = make_env()
        env.reset()
        for agent in env.possible_agents:
            sp = env.action_space(agent)
            self.assertIsInstance(sp, spaces.Discrete)
            self.assertEqual(sp.n, 3)

    def test_step_returns_five_dicts(self):
        env = make_env()
        env.reset()
        result = env.step({"agent_0": 0})
        self.assertEqual(len(result), 5)  # obs, rew, terms, truncs, infos

    def test_inherits_from_multi_agent_env(self):
        from environment.multi_agent_env import MultiAgentBoxPushEnv

        env = make_env()
        self.assertIsInstance(env, MultiAgentBoxPushEnv)


class TestDeterministicMove(unittest.TestCase):
    """With move_success_prob=1.0 the env should behave like the deterministic env."""

    def setUp(self):
        # agent_0 at (2,1) facing down, nothing blocking below
        self.env = make_env(move_p=1.0, push_p=1.0)
        self.env.reset()

    def test_rotate_left_does_not_move_agent(self):
        pos_before = self.env.agent_positions["agent_0"]
        self.env.step({"agent_0": 0})  # rotate left
        self.assertEqual(self.env.agent_positions["agent_0"], pos_before)

    def test_rotate_right_does_not_move_agent(self):
        pos_before = self.env.agent_positions["agent_0"]
        self.env.step({"agent_0": 1})  # rotate right
        self.assertEqual(self.env.agent_positions["agent_0"], pos_before)

    def test_forward_into_empty_moves_agent(self):
        # agent_0 at (2,1) facing down, box at (2,2), so it pushes first
        # Push box to (2,3) so (2,2) is empty, then agent can move forward
        self.env.step({"agent_0": 2})   # push box: agent → (2,2), box → (2,3)
        pos = self.env.agent_positions["agent_0"]
        self.assertEqual(pos, (2, 2))

    def test_forward_blocked_by_wall_stays_put(self):
        # Rotate 180°: agent faces up (toward wall at y=0)
        self.env.step({"agent_0": 1})  # now facing left
        self.env.step({"agent_0": 1})  # now facing up
        pos_before = self.env.agent_positions["agent_0"]
        cell_above = self.env.core_env.grid.get(pos_before[0], pos_before[1] - 1)
        if cell_above is not None and cell_above.type == "wall":
            self.env.step({"agent_0": 2})
            self.assertEqual(self.env.agent_positions["agent_0"], pos_before)


class TestPushSmallBox(unittest.TestCase):
    """Small-box push stochasticity."""

    MAP_WITH_SPACE = [
        "WWWWW",
        "W A W",
        "W B W",
        "W   W",
        "WWWWW",
    ]

    def test_push_succeeds_with_prob_1(self):
        env = make_env(push_p=1.0, ascii_map=self.MAP_WITH_SPACE)
        env.reset()
        # agent_0 at (2,1) facing down; box at (2,2); space at (2,3)
        env.step({"agent_0": 2})
        self.assertEqual(env.agent_positions["agent_0"], (2, 2))
        self.assertEqual(env.core_env.grid.get(2, 3).type, "box")

    def test_push_fails_with_prob_0(self):
        env = make_env(push_p=0.0, ascii_map=self.MAP_WITH_SPACE)
        env.reset()
        # push always fails → agent stays, box stays
        env.step({"agent_0": 2})
        self.assertEqual(env.agent_positions["agent_0"], (2, 1))
        self.assertEqual(env.core_env.grid.get(2, 2).type, "box")

    def test_push_blocked_by_wall_is_always_noop(self):
        # Box at (2,2), agent at (2,1) facing down, WALL at (2,3) → push impossible
        MAP_WALL_BEHIND = [
            "WWWWW",
            "W A W",
            "W B W",
            "WWWWW",
        ]
        env = make_env(push_p=1.0, ascii_map=MAP_WALL_BEHIND)
        env.reset()
        # Even with push_p=1.0, wall behind box blocks the push
        env.step({"agent_0": 2})
        self.assertEqual(env.agent_positions["agent_0"], (2, 1))
        self.assertEqual(env.core_env.grid.get(2, 2).type, "box")


class TestMoveStochasticity(unittest.TestCase):
    """Verify directional deviation logic."""

    # Open map: agent at centre, nothing around
    OPEN_MAP = [
        "WWWWWWW",
        "W     W",
        "W     W",
        "W  A  W",
        "W     W",
        "W     W",
        "WWWWWWW",
    ]

    def test_move_success_prob_0_always_deviates(self):
        """With move_success_prob=0.0 the agent NEVER goes in the intended direction."""
        env = make_env(move_p=0.0, ascii_map=self.OPEN_MAP)
        env.reset()
        start_pos = env.agent_positions["agent_0"]
        # agent_0 faces down (dir=1) → intended fwd is (col, row+1)
        intended_fwd = (start_pos[0], start_pos[1] + 1)

        for _ in range(20):
            env.reset()
            env.step({"agent_0": 2})
            new_pos = env.agent_positions["agent_0"]
            self.assertNotEqual(
                new_pos,
                intended_fwd,
                "Agent moved in intended direction despite move_success_prob=0",
            )

    def test_move_success_prob_1_always_intended(self):
        """With move_success_prob=1.0 the agent ALWAYS goes in the intended direction."""
        env = make_env(move_p=1.0, ascii_map=self.OPEN_MAP)
        env.reset()
        start_pos = env.agent_positions["agent_0"]
        intended_fwd = (start_pos[0], start_pos[1] + 1)

        for _ in range(20):
            env.reset()
            env.step({"agent_0": 2})
            self.assertEqual(env.agent_positions["agent_0"], intended_fwd)

    def test_deviation_blocked_by_wall_stays_put(self):
        """Deviated movement into a wall leaves the agent in place (no error)."""
        # Corridor: agent at (3,2) faces down (intended fwd = (3,3) = empty).
        # Left/right perpendicular cells (2,2) and (4,2) are walls.
        # Row 1 is fully open so MiniGrid's dummy agent_pos=(1,1) assertion passes.
        MAP_CORRIDOR = [
            "WWWWWWW",
            "W     W",   # row 1 fully open — required so MiniGrid dummy pos (1,1) is valid
            "WWWAWWW",   # agent at (3,2); walls left (2,2) and right (4,2)
            "W     W",   # (3,3) is empty — forward is passable
            "WWWWWWW",
        ]
        # move_success_prob=0 → agent ALWAYS deviates to a perpendicular cell.
        # Both perpendicular cells are walls → agent must stay put.
        env = make_env(move_p=0.0, ascii_map=MAP_CORRIDOR)
        env.reset()
        start_pos = env.agent_positions["agent_0"]
        for _ in range(10):
            env.reset()
            env.step({"agent_0": 2})
            self.assertEqual(env.agent_positions["agent_0"], start_pos)

    def test_statistical_move_probabilities(self):
        """Over many trials, intended direction is taken ~80% of the time."""
        env = make_env(move_p=0.8, ascii_map=self.OPEN_MAP)
        n_trials = 2000
        intended_count = 0

        for _ in range(n_trials):
            env.reset()
            start = env.agent_positions["agent_0"]
            env.step({"agent_0": 2})  # forward while facing down
            new_pos = env.agent_positions["agent_0"]
            intended = (start[0], start[1] + 1)
            if new_pos == intended:
                intended_count += 1

        observed_rate = intended_count / n_trials
        self.assertAlmostEqual(observed_rate, 0.8, delta=0.05,
                               msg=f"Expected ~0.80, got {observed_rate:.3f}")


class TestTruncation(unittest.TestCase):
    def test_truncation_after_max_steps(self):
        env = StochasticMultiAgentBoxPushEnv(ascii_map=SIMPLE_MAP, max_steps=3)
        env.reset()
        truncated = False
        for _ in range(3):
            _, _, terms, truncs, _ = env.step({"agent_0": 0})
            if any(truncs.values()):
                truncated = True
        self.assertTrue(truncated)


class TestHeavyBoxPush(unittest.TestCase):
    # Heavy-box push requires BOTH agents at the SAME cell facing the same direction.
    # Map: agent_0 at (2,2), agent_1 at (3,2). We collocate agent_1 onto (2,2) and
    # remove it from the grid so the grid state is consistent before the step.
    # Row 1 is kept open so MiniGrid's dummy agent_pos=(1,1) assertion passes.
    MAP_HEAVY = [
        "WWWWWW",
        "W    W",   # row 1 open — required for MiniGrid dummy pos (1,1)
        "W AA W",   # agent_0 at (2,2), agent_1 at (3,2)
        "W C  W",   # heavy box at (2,3)
        "W    W",   # empty target row for pushed box
        "WWWWWW",
    ]

    def _make_collocated_env(self, push_p):
        """
        Return env where both agents are at (2,2) facing down, heavy box at (2,3).
        agent_1 starts at (3,2) and is relocated to (2,2) with proper grid update.
        """
        env = make_env(push_p=push_p, ascii_map=self.MAP_HEAVY)
        env.reset()
        # Remove agent_1 from its original grid position (3,2)
        old_pos = env.agent_positions["agent_1"]
        if env.core_env.grid.get(*old_pos) is env.agent_objects["agent_1"]:
            env.core_env.grid.set(*old_pos, None)
        # Collocate with agent_0 at (2,2) — we do NOT place agent_1 in the grid;
        # step() clears based on agent_positions, so only agent_0's object is cleared.
        env.agent_positions["agent_1"] = (2, 2)
        env.agent_dirs["agent_1"] = 1   # facing down
        env.agent_objects["agent_1"].dir = 1
        return env

    def test_heavy_push_fails_with_prob_0(self):
        env = self._make_collocated_env(push_p=0.0)
        env.step({"agent_0": 2, "agent_1": 2})
        # Push always fails: heavy box must not have moved from (2,3)
        box = env.core_env.grid.get(2, 3)
        self.assertIsNotNone(box)
        self.assertEqual(getattr(box, "box_size", ""), "heavy")
        # Both agents must stay at (2,2)
        self.assertEqual(env.agent_positions["agent_0"], (2, 2))
        self.assertEqual(env.agent_positions["agent_1"], (2, 2))

    def test_single_agent_cannot_push_heavy_box(self):
        env = self._make_collocated_env(push_p=1.0)
        # Only agent_0 pushes; agent_1 rotates left → not a joint push
        env.step({"agent_0": 2, "agent_1": 0})
        box = env.core_env.grid.get(2, 3)
        self.assertIsNotNone(box)
        self.assertEqual(getattr(box, "box_size", ""), "heavy")

    def test_statistical_heavy_push_probability(self):
        """With push_success_prob=0.8, joint push from same cell succeeds ~80% of runs."""
        n_trials = 500
        successes = 0
        for _ in range(n_trials):
            env = self._make_collocated_env(push_p=0.8)
            env.step({"agent_0": 2, "agent_1": 2})
            # Success: box moved to (2,4)
            if env.core_env.grid.get(2, 4) is not None:
                successes += 1
        rate = successes / n_trials
        self.assertAlmostEqual(rate, 0.8, delta=0.06,
                               msg=f"Expected ~0.80, got {rate:.3f}")


class TestSimultaneousCollisions(unittest.TestCase):
    """Regression tests for the Assignment-4 box-deletion bug: two agents'
    simultaneous moves/pushes landing on the same cell must be resolved
    consistently instead of one silently overwriting the other."""

    # agent_0 sits directly above the cell a box will be pushed onto by
    # agent_1, and both act in the same joint step: agent_0 walks forward
    # onto (3,3) at the exact moment agent_1 pushes the box from (2,3) to
    # (3,3). Both destinations coincide — this must cancel both actions,
    # not silently delete the box.
    MAP_COLLISION = [
        "WWWWWW",
        "W    W",   # row 1 open — required for MiniGrid dummy pos (1,1)
        "W  A W",   # agent_0 at (3,2), default facing down → forward (3,3)
        "WAB  W",   # agent_1 at (1,3), box at (2,3); (3,3) is open
        "W    W",
        "WWWWWW",
    ]

    def test_agent_move_into_box_push_destination_is_cancelled(self):
        """agent_0 tries to walk onto (3,3) at the same moment agent_1
        pushes the box from (2,3) to (3,3) — both must be cancelled, and
        the box must NOT be deleted."""
        env = make_env(move_p=1.0, push_p=1.0, ascii_map=self.MAP_COLLISION)
        env.reset()
        # agent_0 default dir=1 (down) → forward = (3,3): the collision cell.
        # agent_1 needs dir=0 (right) → forward = (2,3): the box.
        env.agent_dirs["agent_1"] = 0
        env.agent_objects["agent_1"].dir = 0

        boxes_before = self._count_boxes(env)
        self.assertEqual(boxes_before, 1)

        env.step({"agent_0": 2, "agent_1": 2})  # both move forward

        boxes_after = self._count_boxes(env)
        self.assertEqual(
            boxes_after, boxes_before,
            "Box was lost — the simultaneous-collision bug has regressed."
        )
        # Collision → both actions cancelled, both agents stay put.
        self.assertEqual(env.agent_positions["agent_0"], (3, 2))
        self.assertEqual(env.agent_positions["agent_1"], (1, 3))
        # The box must still be exactly where it started.
        self.assertEqual(env.core_env.grid.get(2, 3).type, "box")

    @staticmethod
    def _count_boxes(env):
        n = 0
        for y in range(env.height):
            for x in range(env.width):
                cell = env.core_env.grid.get(x, y)
                if cell is not None and getattr(cell, "type", "") == "box":
                    n += 1
        return n

    def test_two_agents_pushing_same_small_box_are_both_cancelled(self):
        """Two agents adjacent to the same small box, both attempting to
        push it (in different directions) in the same step: neither push
        may apply, and the box must not duplicate or vanish."""
        MAP = [
            "WWWWW",
            "W   W",
            "WAB W",
            "W A W",
            "WWWWW",
        ]
        env = make_env(move_p=1.0, push_p=1.0, ascii_map=MAP)
        env.reset()
        # agent_0 at (1,2) facing right → forward = (2,2) = box
        env.agent_dirs["agent_0"] = 0
        env.agent_objects["agent_0"].dir = 0
        # agent_1 at (2,3) facing up → forward = (2,2) = box (same box!)
        env.agent_dirs["agent_1"] = 3
        env.agent_objects["agent_1"].dir = 3

        boxes_before = self._count_boxes(env)
        env.step({"agent_0": 2, "agent_1": 2})
        boxes_after = self._count_boxes(env)

        self.assertEqual(boxes_after, boxes_before, "Box count changed — duplication or deletion.")
        self.assertEqual(env.core_env.grid.get(2, 2).type, "box",
                         "Contested box must stay at its original cell.")
        self.assertEqual(env.agent_positions["agent_0"], (1, 2))
        self.assertEqual(env.agent_positions["agent_1"], (2, 3))


if __name__ == "__main__":
    unittest.main()
