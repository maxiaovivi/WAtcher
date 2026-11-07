import numpy as np
import pytest

from safe_wam.contracts import Decision, RiskScores
from safe_wam.simulation_loop import run_episode
from safe_wam.simulation_pipeline import GuardedSimulationPolicy


class Env:
    def __init__(self, limit=20, success_after=None):
        self.take_action_cnt=0;self.step_lim=limit;self.eval_success=False
        self.success_after=success_after;self.executed=[];self.trace=[]
    def get_obs(self):
        self.trace.append("observe")
        return {"observation":{c+"_camera":{"rgb":np.zeros((4,4,3),np.uint8)} for c in ("head","left","right")},
                "joint_action":{"vector":np.zeros(14,np.float32)}}
    def take_action(self, action, *, action_type):
        assert action_type == "qpos" and self.trace.count("score") > 0
        self.trace.append("execute");self.executed.append(action.copy());self.take_action_cnt+=1
        if self.success_after is not None and self.take_action_cnt>=self.success_after:self.eval_success=True


def controller(env, risk=.1, error=False):
    class Policy:
        def predict(self, obs, *, instruction):
            env.trace.append("predict")
            a=np.full((16,14),.2,np.float32)
            a[:,0]=np.arange(16,dtype=np.float32)/10
            return a
    class Head:
        def score(self, obs, a):
            env.trace.append("score")
            if error:raise RuntimeError("unavailable")
            return RiskScores(np.full(16,risk),risk,risk,.5)
    return GuardedSimulationPolicy(Head(),Policy())


def test_score_precedes_exact_prefix_and_replans_at_each_chunk():
    e=Env()
    r=run_episode(e,controller(e),"task",execute_steps=3,max_actions=5)
    assert e.trace==["observe","predict","score","execute","execute","execute","observe","predict","score","execute","execute"]
    np.testing.assert_array_equal(np.array(e.executed)[:,0],np.array([0,.1,.2,0,.1],np.float32))
    assert r["simulator_actions"]==5 and r["risk_evaluations"]==2
    assert [x["executed_actions"] for x in r["events"]]==[3,2]
    assert not r["task_success"] and not r["physical_robot_commands_sent"]


def test_guard_rejection_sends_no_simulator_actions():
    e=Env();r=run_episode(e,controller(e,.9),"task")
    assert r["status"]=="RISK_REJECTED" and not e.executed
    assert r["events"][0]["risk"]["overall"]==.9


def test_monitor_retains_reject_label_while_logging_sim_execution():
    e=Env(limit=2);r=run_episode(e,controller(e,.9),"task",mode="monitor")
    event=r["events"][0]
    assert event["status"]=="REJECT" and event["allowed"] is False
    assert event["high_risk_executed_in_monitor"] is True and event["executed_actions"]==2
    assert r["task_success"] is False and r["status"]=="TASK_HORIZON"


@pytest.mark.parametrize("mode",["guard","monitor"])
def test_model_error_stops_even_monitor(mode):
    e=Env();r=run_episode(e,controller(e,error=True),"task",mode=mode)
    assert r["status"]=="SCORING_ERROR" and not e.executed


def test_task_success_is_independent_of_high_risk_monitor_decision():
    e=Env(success_after=1);r=run_episode(e,controller(e,.9),"task",mode="monitor")
    assert r["task_success"] is True and r["simulator_actions"]==1
    assert r["events"][0]["status"]=="REJECT"


def test_existing_success_does_not_generate_or_score():
    e=Env();e.eval_success=True
    r=run_episode(e,controller(e),"task")
    assert r["status"]=="TASK_SUCCESS" and not e.trace and r["risk_evaluations"]==0


@pytest.mark.parametrize("kwargs",[{"mode":"real"},{"execute_steps":0},{"execute_steps":17},{"max_actions":0}])
def test_invalid_execution_configuration_sends_nothing(kwargs):
    e=Env()
    with pytest.raises(ValueError):run_episode(e,controller(e),"task",**kwargs)
    assert not e.trace


def test_simulator_counter_failure_stops_after_first_action():
    e=Env()
    def broken(action,**kwargs):e.executed.append(action)
    e.take_action=broken
    with pytest.raises(RuntimeError,match="advance"):
        run_episode(e,controller(e),"task")
    assert len(e.executed)==1


@pytest.mark.parametrize("mode", ["guard", "monitor"])
@pytest.mark.parametrize("fault", ["nan_risk", "wrong_shape", "allow_high_risk", "reject_low_risk"])
def test_foreign_controller_cannot_bypass_execution_validation(mode, fault):
    e = Env()
    a = np.full((16, 14), .2, np.float32)
    p, status = .9, "REJECT"
    if fault == "nan_risk": p = np.nan
    if fault == "wrong_shape": a = np.zeros((16, 7), np.float32)
    if fault == "allow_high_risk": status = "ALLOW"
    if fault == "reject_low_risk": p = .1
    d = Decision(status, "foreign", a, RiskScores(np.full(16, p), p, p, .5))
    class Foreign:
        def evaluate(self, obs, instruction): return d
    with pytest.raises(ValueError):
        run_episode(e, Foreign(), "task", mode=mode)
    assert e.executed == [] and e.take_action_cnt == 0
    assert e.trace == ["observe"]
