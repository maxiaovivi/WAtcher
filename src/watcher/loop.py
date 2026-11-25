"""Closed loop: observe -> generate -> score -> execute a prefix of the scored chunk -> observe again."""
import numpy as np

from .contracts import HORIZON, check_actions
from .gate import check_risk


def run_episode(task, gate, instruction, *, execute_steps=HORIZON, max_actions=None, mode="guard",
                on_event=None):
    """Run one RoboTwin episode and return a summary dict with one event per scored chunk.

    mode='guard' stops the episode at the first REJECT without executing it. mode='monitor' is a
    simulation-only comparison that executes rejected chunks too (marked `high_risk_executed`).
    Any ERROR decision stops the episode in both modes.
    """
    if mode not in ("guard", "monitor"):
        raise ValueError("mode must be 'guard' or 'monitor'")
    if not isinstance(execute_steps, int) or not 1 <= execute_steps <= HORIZON:
        raise ValueError(f"execute_steps must be an int in 1..{HORIZON}")
    if max_actions is not None and (not isinstance(max_actions, int) or max_actions < 1):
        raise ValueError("max_actions must be a positive int")
    start, horizon = int(task.take_action_cnt), int(task.step_lim)
    limit = horizon if max_actions is None else min(horizon, start + max_actions)
    events, status = [], "ACTION_LIMIT"

    def finish(event):
        events.append(event)
        if on_event:
            on_event(event)

    while task.take_action_cnt < limit and not task.eval_success:
        decision = gate.evaluate(task.get_obs(), instruction)
        event = dict(chunk=len(events), actions_before=int(task.take_action_cnt), **decision.to_dict(),
                     executed=0, high_risk_executed=False)
        if decision.status not in ("ALLOW", "REJECT"):
            status = "ERROR"
            finish(event)
            break
        # Re-validate whatever the gate returned: nothing unscored or mislabeled reaches the simulator.
        actions, risk = check_actions(decision.actions), check_risk(decision.risk)
        if decision.allowed != (risk.overall < risk.threshold):
            raise ValueError("Decision status contradicts its risk score")
        if not decision.allowed and mode == "guard":
            status = "RISK_REJECTED"
            finish(event)
            break
        event["high_risk_executed"] = not decision.allowed
        for action in actions[:execute_steps]:
            if task.take_action_cnt >= limit or task.eval_success:
                break
            before = task.take_action_cnt
            task.take_action(np.array(action), action_type="qpos")
            if task.take_action_cnt != before + 1:
                raise RuntimeError("Simulator did not advance by exactly one action")
            event["executed"] += 1
        finish(event)

    success = bool(task.eval_success)
    if success:
        status = "TASK_SUCCESS"
    elif status == "ACTION_LIMIT" and task.take_action_cnt >= horizon:
        status = "TASK_HORIZON"
    return dict(status=status, task_success=success, actions=int(task.take_action_cnt) - start,
                chunks=len(events), mode=mode, execute_steps=execute_steps, horizon=horizon, events=events)
