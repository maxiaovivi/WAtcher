import numpy as np
import pytest

from safe_wam.contracts import RiskScores, SimulationObservation, validate_simulation_observation
from safe_wam.simulation_pipeline import GuardedSimulationPolicy


def native():
    return {"observation": {c + "_camera": {"rgb": np.full((8, 9, 3), i, np.uint8)}
                            for i, c in enumerate(("head", "left", "right"))},
            "joint_action": {"vector": np.zeros(14, np.float32)}}


def candidate():
    a = np.arange(224, dtype=np.float32).reshape(16, 14) / 100
    a[:, [6, 13]] = .5
    return a


class Scorer:
    def __init__(self, step=.1, post=.2, overall=None):
        self.calls = []
        self.risk = RiskScores(np.full(16, step), post, max(step, post) if overall is None else overall, .5)

    def score(self, obs, actions):
        self.calls.append((obs, actions.copy()))
        return self.risk


def test_external_candidate_bypasses_policy_and_keeps_named_views():
    class Forbidden:
        def predict(self, *a, **k): pytest.fail("External actions must bypass WAM")
    s, a = Scorer(), candidate()
    d = GuardedSimulationPolicy(s, Forbidden()).evaluate(native(), "stack bowls", a)
    assert d.allowed and len(s.calls) == 1
    for i, camera in enumerate(("head", "left", "right")):
        assert np.all(s.calls[0][0].images[camera] == i)
    np.testing.assert_array_equal(s.calls[0][1], a)
    a[:] = 0
    np.testing.assert_array_equal(d.approved_actions, candidate())
    assert not d.approved_actions.flags.writeable
    assert d.to_dict()["physical_robot_commands_sent"] is False


def test_policy_receives_snapshot_copy_and_scorer_sees_original():
    original = native()
    class Policy:
        def predict(self, obs, *, instruction):
            assert instruction == "place cup"
            obs["joint_action"]["vector"][:] = 123
            obs["observation"]["head_camera"]["rgb"][:] = 255
            original["joint_action"]["vector"][:] = 7
            return candidate()
    s = Scorer()
    assert GuardedSimulationPolicy(s, Policy()).evaluate(original, "place cup").allowed
    assert np.all(s.calls[0][0].current_target14 == 0)
    assert np.all(s.calls[0][0].images["head"] == 0)


@pytest.mark.parametrize("step,post,overall,status", [(.499,.1,None,"ALLOW"),(.5,.1,None,"REJECT"),
    (.1,.9,None,"REJECT"),(.5,.1,.4999995,"REJECT"),(.1,.1,.01,"SCORING_ERROR"),
    (np.nan,.1,None,"SCORING_ERROR"),(np.inf,.1,None,"SCORING_ERROR")])
def test_risk_max_and_fail_closed(step, post, overall, status):
    d = GuardedSimulationPolicy(Scorer(step, post, overall)).evaluate(native(), "task", candidate())
    assert d.status == status
    if status != "ALLOW": assert d.approved_actions is None


@pytest.mark.parametrize("value", [np.zeros((16,7)),np.zeros((8,14)),np.full((16,14),np.nan),np.ones((16,14),complex)*1j])
def test_unsupported_candidate_never_scores(value):
    s = Scorer()
    d = GuardedSimulationPolicy(s).evaluate(native(), "task", value)
    assert d.status == "INVALID_INPUT" and not s.calls


def test_two_grippers_clip_before_score_without_changing_arm_targets():
    raw = candidate(); raw[0,6] = -.1; raw[2,13] = 1.01
    original = raw.copy(); s = Scorer()
    assert GuardedSimulationPolicy(s).evaluate(native(), "task", raw).status == "INVALID_INPUT"
    assert not s.calls
    d = GuardedSimulationPolicy(s,gripper_mode="clip").evaluate(native(), "task", raw)
    expected = original.copy();expected[0,6]=0;expected[2,13]=1
    np.testing.assert_array_equal(s.calls[0][1],expected)
    np.testing.assert_array_equal(d.approved_actions,expected)
    np.testing.assert_array_equal(raw,original)
    assert d.gripper_values_clipped == 2


def test_invalid_observation_and_instruction_do_not_score():
    s = Scorer(); bad = native(); del bad["observation"]["right_camera"]
    assert GuardedSimulationPolicy(s).evaluate(bad,"task",candidate()).status == "INVALID_INPUT"
    assert GuardedSimulationPolicy(s).evaluate(native()," ",candidate()).status == "INVALID_INPUT"
    assert not s.calls


def test_future_observation_timestamp_is_invalid():
    obs = SimulationObservation({k: np.zeros((2,2,3),np.uint8) for k in ("head","left","right")},np.zeros(14),.1,0)
    with pytest.raises(ValueError): validate_simulation_observation(obs)


def test_mutating_scorer_fails_closed():
    class Bad(Scorer):
        def score(self, obs, a): a[0,0]=9;return self.risk
    d = GuardedSimulationPolicy(Bad()).evaluate(native(),"task",candidate())
    assert d.status == "SCORING_ERROR" and d.approved_actions is None
    np.testing.assert_array_equal(d.candidate_actions,candidate())
