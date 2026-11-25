import numpy as np
import pytest

from watcher import Decision, Gate, RiskScores, run_episode
from watcher.contracts import CAMERAS


class Task:
    """Minimal stand-in for a RoboTwin task."""

    def __init__(self, limit=20, success_after=None):
        self.take_action_cnt, self.step_lim, self.eval_success = 0, limit, False
        self.success_after, self.executed, self.trace = success_after, [], []

    def get_obs(self):
        self.trace.append("observe")
        return {"observation": {name: {"rgb": np.zeros((4, 4, 3), np.uint8)} for name in CAMERAS},
                "joint_action": {"vector": np.zeros(14, np.float32)}}

    def take_action(self, action, *, action_type):
        assert action_type == "qpos" and "score" in self.trace
        self.trace.append("execute")
        self.executed.append(action.copy())
        self.take_action_cnt += 1
        if self.success_after is not None and self.take_action_cnt >= self.success_after:
            self.eval_success = True


def gate(task, risk=0.1, error=False):
    class Policy:
        def predict(self, obs, *, instruction):
            task.trace.append("predict")
            actions = np.full((16, 14), 0.2, np.float32)
            actions[:, 0] = np.arange(16) / 10
            return actions

    class Head:
        def score(self, images, state, actions):
            task.trace.append("score")
            if error:
                raise RuntimeError("unavailable")
            return RiskScores(np.full(16, risk), risk, risk, 0.5)
    return Gate(Head(), Policy())


def test_each_chunk_is_scored_before_its_prefix_is_executed():
    task = Task()
    result = run_episode(task, gate(task), "task", execute_steps=3, max_actions=5)
    assert task.trace == ["observe", "predict", "score", "execute", "execute", "execute",
                          "observe", "predict", "score", "execute", "execute"]
    np.testing.assert_array_equal(np.array(task.executed)[:, 0], np.float32([0, 0.1, 0.2, 0, 0.1]))
    assert (result["status"], result["actions"], result["chunks"]) == ("ACTION_LIMIT", 5, 2)
    assert [event["executed"] for event in result["events"]] == [3, 2]


def test_guard_rejection_executes_nothing():
    task = Task()
    result = run_episode(task, gate(task, 0.9), "task")
    assert result["status"] == "RISK_REJECTED" and not task.executed
    assert result["events"][0]["risk"]["overall"] == 0.9


def test_monitor_executes_rejected_chunks_and_marks_them():
    task = Task(limit=2)
    result = run_episode(task, gate(task, 0.9), "task", mode="monitor")
    event = result["events"][0]
    assert event["status"] == "REJECT" and event["high_risk_executed"] and event["executed"] == 2
    assert result["status"] == "TASK_HORIZON" and not result["task_success"]


@pytest.mark.parametrize("mode", ["guard", "monitor"])
def test_scoring_error_stops_both_modes(mode):
    task = Task()
    result = run_episode(task, gate(task, error=True), "task", mode=mode)
    assert result["status"] == "ERROR" and not task.executed


def test_success_is_reported_even_if_the_chunk_was_high_risk():
    task = Task(success_after=1)
    result = run_episode(task, gate(task, 0.9), "task", mode="monitor")
    assert result["status"] == "TASK_SUCCESS" and result["actions"] == 1


def test_already_successful_task_is_not_scored():
    task = Task()
    task.eval_success = True
    result = run_episode(task, gate(task), "task")
    assert result["status"] == "TASK_SUCCESS" and not task.trace and result["chunks"] == 0


@pytest.mark.parametrize("kwargs", [{"mode": "real"}, {"execute_steps": 0}, {"execute_steps": 17}, {"max_actions": 0}])
def test_invalid_settings_send_nothing(kwargs):
    task = Task()
    with pytest.raises(ValueError):
        run_episode(task, gate(task), "task", **kwargs)
    assert not task.trace


def test_simulator_that_does_not_advance_stops_the_episode():
    task = Task()
    task.take_action = lambda action, **kwargs: task.executed.append(action)
    with pytest.raises(RuntimeError, match="advance"):
        run_episode(task, gate(task), "task")
    assert len(task.executed) == 1


@pytest.mark.parametrize("mode", ["guard", "monitor"])
@pytest.mark.parametrize("fault", ["nan_risk", "wrong_shape", "allow_high_risk", "reject_low_risk"])
def test_a_foreign_gate_cannot_bypass_validation(mode, fault):
    actions, risk, status = np.full((16, 14), 0.2, np.float32), 0.9, "REJECT"
    if fault == "nan_risk":
        risk = np.nan
    if fault == "wrong_shape":
        actions = np.zeros((16, 7), np.float32)
    if fault == "allow_high_risk":
        status = "ALLOW"
    if fault == "reject_low_risk":
        risk = 0.1
    decision = Decision(status, "foreign", actions, RiskScores(np.full(16, risk), risk, risk, 0.5))

    class Foreign:
        def evaluate(self, obs, instruction):
            return decision
    task = Task()
    with pytest.raises(ValueError):
        run_episode(task, Foreign(), "task", mode=mode)
    assert task.trace == ["observe"] and not task.executed
