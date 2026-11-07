from dataclasses import replace
import socket

import numpy as np
import pytest

from safe_wam.contracts import CAMERAS, Frame, Observation, RiskScores
from safe_wam.pipeline import GuardedPolicy


def observation():
    return Observation(tuple(Frame({c: np.full((8, 10, 3), i, np.uint8) for c in CAMERAS}, t)
                             for i, t in enumerate((-.3, -.2, -.1, 0.))),
                       np.array([.1, .2, .3, .4, .5, .6, .7]), 0.)


def actions():
    a = np.arange(112, dtype=np.float32).reshape(16, 7) / 1000
    a[:, 6] = np.arange(16) % 2
    return a


class Scorer:
    def __init__(self, step=.1, post=.2, overall=None, threshold=.5):
        self.result = RiskScores(np.full(16, step), post, max(step, post) if overall is None else overall, threshold)
        self.calls = []

    def score(self, obs, candidate):
        self.calls.append((obs, candidate.copy()))
        return self.result


def test_external_actions_bypass_policy_and_preserve_physical_candidate(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("Network/hardware must not be opened"))
    class UnusedPolicy:
        def predict(self, obs):
            pytest.fail("External actions must bypass the policy")
    original = actions()
    scorer = Scorer()
    result = GuardedPolicy(scorer, UnusedPolicy()).evaluate(observation(), original)
    assert result.allowed
    np.testing.assert_array_equal(scorer.calls[0][1], original)
    original[:] = 0
    np.testing.assert_array_equal(result.approved_actions, actions())
    assert not result.approved_actions.flags.writeable
    assert result.to_dict()["execution_authorized"] is False
    assert result.to_dict()["robot_commands_sent"] is False
    assert result.to_dict()["risk"]["step_scores_validated"] is False


@pytest.mark.parametrize("risk,status", [(.499, "ALLOW"), (.5, "REJECT"), (.501, "REJECT")])
def test_inclusive_threshold(risk, status):
    assert GuardedPolicy(Scorer(post=risk)).evaluate(observation(), actions()).status == status


@pytest.mark.parametrize("step,post", [(.9, .1), (.1, .9)])
def test_step_or_post_danger_rejects_without_fallback(step, post):
    result = GuardedPolicy(Scorer(step, post)).evaluate(observation(), actions())
    assert result.status == "REJECT" and result.approved_actions is None
    np.testing.assert_array_equal(result.candidate_actions, actions())


def test_custom_threshold():
    assert GuardedPolicy(Scorer(post=.3, threshold=.3)).evaluate(observation(), actions()).status == "REJECT"


@pytest.mark.parametrize("scorer", [Scorer(post=float("nan")), Scorer(step=float("inf")),
                                   Scorer(overall=.01), Scorer(threshold=1.1)])
def test_invalid_risk_fails_closed(scorer):
    result = GuardedPolicy(scorer).evaluate(observation(), actions())
    assert result.status == "SCORING_ERROR" and result.approved_actions is None


def test_rounding_tolerance_must_not_allow_a_step_at_threshold():
    result = GuardedPolicy(Scorer(step=.5, post=.1, overall=.4999995)).evaluate(observation(), actions())
    assert not result.allowed


def test_complex_risk_is_not_silently_cast_to_real():
    scorer = Scorer()
    scorer.result = RiskScores(np.full(16, .1 + 2j), .2, .2, .5)
    result = GuardedPolicy(scorer).evaluate(observation(), actions())
    assert result.status == "SCORING_ERROR" and not result.allowed


@pytest.mark.parametrize("change", [
    lambda o: replace(o, frames=o.frames[:3]),
    lambda o: replace(o, frames=o.frames[:-1] + (replace(o.frames[-1], timestamp_s=.1),)),
    lambda o: replace(o, frames=tuple(reversed(o.frames))),
    lambda o: replace(o, state_timestamp_s=.1),
    lambda o: replace(o, state=np.zeros(14)),
    lambda o: replace(o, state=np.full(7, np.nan)),
    lambda o: replace(o, state=o.state.astype(complex) + 1j),
    lambda o: replace(o, frames=(replace(o.frames[0], images={c: np.zeros((8, 8, 3), np.float32) for c in CAMERAS}),) + o.frames[1:]),
    lambda o: replace(o, frames=(replace(o.frames[0], images={"external": np.zeros((8, 8, 3), np.uint8)}),) + o.frames[1:]),
])
def test_bad_observation_rejected_before_models(change):
    class Never:
        def predict(self, obs): pytest.fail("Policy called for invalid input")
        def score(self, obs, a): pytest.fail("Scorer called for invalid input")
    result = GuardedPolicy(Never(), Never()).evaluate(change(observation()))
    assert result.status == "INVALID_INPUT"


@pytest.mark.parametrize("a", [np.zeros((8, 7)), np.zeros((16, 14)), np.full((16, 7), np.nan),
                               np.full((16, 7), 1e100), np.ones((16, 7), complex) * 1j])
def test_invalid_actions_never_score(a):
    scorer = Scorer()
    result = GuardedPolicy(scorer).evaluate(observation(), a)
    assert result.status == "INVALID_INPUT" and not scorer.calls


def test_model_errors_do_not_authorize_execution():
    class Broken:
        def predict(self, obs): raise RuntimeError("policy unavailable")
        def score(self, obs, a): raise RuntimeError("scorer unavailable")
    assert GuardedPolicy(Scorer(), Broken()).evaluate(observation()).status == "POLICY_ERROR"
    assert GuardedPolicy(Broken()).evaluate(observation(), actions()).status == "SCORING_ERROR"


def test_scorer_cannot_change_candidate():
    class Mutating(Scorer):
        def score(self, obs, a):
            a[0, 0] = 4
            return self.result
    result = GuardedPolicy(Mutating()).evaluate(observation(), actions())
    assert result.status == "SCORING_ERROR"
    np.testing.assert_array_equal(result.candidate_actions, actions())


def test_policy_and_scorer_share_isolated_snapshot():
    source = observation()
    class Policy:
        def predict(self, snap):
            self.snapshot = snap
            source.state[:] = 0
            source.frames[0].images["external"][:] = 255
            return actions()
    policy, scorer = Policy(), Scorer()
    assert GuardedPolicy(scorer, policy).evaluate(source).allowed
    snap = scorer.calls[0][0]
    assert snap is policy.snapshot and snap.state[6] == .7
    assert np.all(snap.frames[0].images["external"] == 0)


@pytest.mark.parametrize("generated", [False, True])
def test_gripper_clip_is_explicit_and_precedes_scoring(generated):
    raw = actions()
    raw[0, 6], raw[1, 6], raw[2, 6] = 1.0052298, -.03, .37
    saved = raw.copy()
    class Policy:
        def predict(self, obs): return raw
    strict_scorer = Scorer()
    strict = GuardedPolicy(strict_scorer, Policy()).evaluate(observation(), None if generated else raw)
    assert strict.status == ("POLICY_ERROR" if generated else "INVALID_INPUT")
    assert strict.approved_actions is None and not strict_scorer.calls
    scorer = Scorer()
    result = GuardedPolicy(scorer, Policy(), gripper_mode="clip").evaluate(
        observation(), None if generated else raw)
    assert result.allowed and len(scorer.calls) == 1
    expected = saved.copy()
    expected[0, 6], expected[1, 6] = 1, 0
    np.testing.assert_array_equal(scorer.calls[0][1], expected)
    np.testing.assert_array_equal(result.approved_actions, expected)
    np.testing.assert_array_equal(result.approved_actions[:, :6], saved[:, :6])
    np.testing.assert_array_equal(raw, saved)
    assert result.to_dict()["action_processing"] == {"gripper_mode": "clip", "gripper_values_clipped": 2}


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_gripper_clip_does_not_hide_nonfinite_input(value):
    raw = actions()
    raw[0, 6] = value
    scorer = Scorer()
    result = GuardedPolicy(scorer, gripper_mode="clip").evaluate(observation(), raw)
    assert result.status == "INVALID_INPUT" and not scorer.calls and result.approved_actions is None


def test_gripper_clip_does_not_bypass_risk_rejection():
    raw = actions()
    raw[0, 6] = 1.0052298
    result = GuardedPolicy(Scorer(post=.9), gripper_mode="clip").evaluate(observation(), raw)
    assert result.status == "REJECT" and result.approved_actions is None
    assert result.candidate_actions[0, 6] == 1
    assert result.gripper_values_clipped == 1


@pytest.mark.parametrize("mode", ["auto", "CLIP", "", None])
def test_invalid_gripper_mode_is_not_silently_selected(mode):
    with pytest.raises(ValueError, match="gripper_mode"):
        GuardedPolicy(Scorer(), gripper_mode=mode)
