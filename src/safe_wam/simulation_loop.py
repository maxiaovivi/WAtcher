"""RoboTwin-only synchronous task execution and pre-action risk evaluation."""
import numpy as np
from .contracts import validate_actions, validate_risk


def run_episode(env, controller, instruction, *, execute_steps=16, max_actions=None, mode="guard", on_decision=None):
    if mode not in ("guard", "monitor"):
        raise ValueError("mode must be guard or monitor")
    if not isinstance(execute_steps, int) or not 1 <= execute_steps <= 16:
        raise ValueError("execute_steps must lie in 1..16")
    if max_actions is not None and (not isinstance(max_actions, int) or max_actions < 1):
        raise ValueError("max_actions must be positive")
    start = int(env.take_action_cnt)
    horizon = int(env.step_lim)
    limit = horizon if max_actions is None else min(horizon, start + max_actions)
    events = []
    result = "ACTION_LIMIT"
    while int(env.take_action_cnt) < limit and not bool(env.eval_success):
        native = env.get_obs()
        decision = controller.evaluate(native, instruction)
        event = dict(chunk=len(events), action_count_before=int(env.take_action_cnt),
                     **decision.to_dict(include_actions=False), executed_actions=0,
                     mode=mode, high_risk_executed_in_monitor=False)
        if decision.status not in ("ALLOW", "REJECT"):
            events.append(event); result = decision.status
            if on_decision: on_decision(event)
            break
        if decision.candidate_actions is None or decision.risk is None:
            raise RuntimeError("No scored candidate is present")
        scores = validate_risk(decision.risk)
        candidate = validate_actions(decision.candidate_actions)
        if decision.allowed != (scores.overall < scores.threshold):
            raise ValueError("Decision status contradicts the risk threshold")
        if not decision.allowed and mode == "guard":
            events.append(event); result = "RISK_REJECTED"
            if on_decision: on_decision(event)
            break
        # Monitor mode is a simulation-only comparison; the REJECT label remains intact.
        event["high_risk_executed_in_monitor"] = not decision.allowed
        for action in candidate[:execute_steps]:
            if int(env.take_action_cnt) >= limit or bool(env.eval_success):
                break
            before = int(env.take_action_cnt)
            env.take_action(np.array(action, copy=True), action_type="qpos")
            if int(env.take_action_cnt) != before + 1:
                raise RuntimeError("Simulator did not advance exactly one native action")
            event["executed_actions"] += 1
        events.append(event)
        if on_decision: on_decision(event)
    success = bool(env.eval_success)
    if success:
        result = "TASK_SUCCESS"
    elif result == "ACTION_LIMIT" and int(env.take_action_cnt) >= horizon:
        result = "TASK_HORIZON"
    return dict(status=result, task_success=success, simulator_actions=int(env.take_action_cnt)-start,
                risk_evaluations=len(events), execution_mode=mode, execute_steps=execute_steps,
                official_action_horizon=horizon, events=events, physical_robot_commands_sent=False)
