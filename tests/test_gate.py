import numpy as np
import pytest

from watcher import Gate, RiskScores
from watcher.contracts import CAMERAS


def native():
    return {"observation": {name: {"rgb": np.full((8, 9, 3), i, np.uint8)} for i, name in enumerate(CAMERAS)},
            "joint_action": {"vector": np.zeros(14, np.float32)}}


def candidate():
    actions = np.arange(224, dtype=np.float32).reshape(16, 14) / 100
    actions[:, [6, 13]] = 0.5
    return actions


class Head:
    def __init__(self, step=0.1, post=0.2, overall=None):
        self.calls = []
        self.risk = RiskScores(np.full(16, step), post, max(step, post) if overall is None else overall, 0.5)

    def score(self, images, state, actions):
        self.calls.append((images, state, actions))
        return self.risk


def test_external_candidate_is_scored_without_the_policy():
    class Forbidden:
        def predict(self, *args, **kwargs):
            pytest.fail("external actions must bypass the policy")
    head, actions = Head(), candidate()
    decision = Gate(head, Forbidden()).evaluate(native(), "stack bowls", actions)
    assert decision.allowed and len(head.calls) == 1
    for i, name in enumerate(CAMERAS):
        assert np.all(head.calls[0][0][name] == i)
    actions[:] = 0  # the decision keeps its own read-only copy
    np.testing.assert_array_equal(decision.actions, candidate())
    assert not decision.actions.flags.writeable


def test_policy_and_head_cannot_change_each_others_inputs():
    original = native()

    class Policy:
        def predict(self, obs, *, instruction):
            assert instruction == "place cup"
            obs["joint_action"]["vector"][:] = 0.3
            obs["observation"]["head_camera"]["rgb"][:] = 255
            return candidate()

    class MutatingHead(Head):
        def score(self, images, state, actions):
            super().score(images, state.copy(), actions.copy())
            actions[0, 0] = 9
            return self.risk
    head = MutatingHead()
    decision = Gate(head, Policy()).evaluate(original, "place cup")
    assert decision.allowed and np.all(head.calls[0][1] == 0) and np.all(head.calls[0][0]["head_camera"] == 0)
    np.testing.assert_array_equal(decision.actions, candidate())
    assert np.all(original["joint_action"]["vector"] == 0)


@pytest.mark.parametrize("step,post,overall,status", [
    (0.499, 0.1, None, "ALLOW"), (0.5, 0.1, None, "REJECT"), (0.1, 0.9, None, "REJECT"),
    (0.5, 0.1, 0.4999995, "REJECT"), (0.1, 0.1, 0.01, "ERROR"), (np.nan, 0.1, None, "ERROR"), (np.inf, 0.1, None, "ERROR")])
def test_threshold_uses_max_of_step_and_post_and_fails_closed(step, post, overall, status):
    decision = Gate(Head(step, post, overall)).evaluate(native(), "task", candidate())
    assert decision.status == status
    assert (decision.risk is not None) == (status != "ERROR")


@pytest.mark.parametrize("actions", [np.zeros((16, 7)), np.zeros((8, 14)), np.full((16, 14), np.nan),
                                     np.ones((16, 14), complex)])
def test_invalid_candidate_is_never_scored(actions):
    head = Head()
    assert Gate(head).evaluate(native(), "task", actions).status == "ERROR" and not head.calls


def test_grippers_are_clipped_before_scoring_and_arms_are_unchanged():
    raw = candidate()
    raw[0, 6], raw[2, 13] = -0.1, 1.01
    head = Head()
    assert Gate(head).evaluate(native(), "task", raw).status == "ERROR" and not head.calls
    decision = Gate(head, gripper_mode="clip").evaluate(native(), "task", raw)
    expected = candidate()
    expected[0, 6], expected[2, 13] = 0, 1
    np.testing.assert_array_equal(head.calls[0][2], expected)
    np.testing.assert_array_equal(decision.actions, expected)
    assert decision.clipped == 2 and raw[0, 6] == np.float32(-0.1)  # the caller's array is untouched


def test_invalid_observation_or_instruction_is_never_scored():
    head, bad = Head(), native()
    del bad["observation"]["right_camera"]
    assert Gate(head).evaluate(bad, "task", candidate()).status == "ERROR"
    assert Gate(head).evaluate(native(), " ", candidate()).status == "ERROR"
    assert Gate(head).evaluate(native(), "task").status == "ERROR"  # no policy and no actions
    assert not head.calls


def test_head_exception_keeps_the_candidate_but_no_risk():
    class Broken(Head):
        def score(self, *args):
            raise RuntimeError("unavailable")
    decision = Gate(Broken()).evaluate(native(), "task", candidate())
    assert decision.status == "ERROR" and decision.risk is None and "unavailable" in decision.reason
