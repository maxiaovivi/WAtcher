"""Create a native RoboTwin 2.0 task (validated at commit c3ddfa8) in the current process."""
from copy import deepcopy
import importlib
import os
from pathlib import Path
import re
import sys


def create_task(root, task_name, *, setting="clean", seed=0, instruction=None, render_device="cuda:0"):
    """Set up `envs/<task_name>.py` the way RoboTwin's evaluation does, on the dual aloha-agilex arms.

    RoboTwin relies on relative paths and global state: call this once per process, before anything
    else from RoboTwin is imported. The episode step limit comes from `_eval_step_limit.yml`.
    """
    import sapien
    import yaml

    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task_name) or not (root / f"envs/{task_name}.py").is_file():
        raise ValueError(f"No RoboTwin task {task_name!r} in {root}")
    if setting not in ("clean", "randomized"):
        raise ValueError("setting must be 'clean' or 'randomized'")
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "policy"), str(root / "description/utils")]

    def read(name):
        return yaml.safe_load((root / "task_config" / name).read_text())
    args = read(f"demo_{setting}.yml")
    if args["embodiment"] != ["aloha-agilex"]:
        raise ValueError("The 14-D action contract needs the dual aloha-agilex embodiment")
    robot = Path(read("_embodiment_config.yml")["aloha-agilex"]["file_path"])
    robot = (robot if robot.is_absolute() else root / robot).resolve(strict=True)
    robot_config = yaml.safe_load((robot / "config.yml").read_text())
    head = read("_camera_config.yml")[args["camera"]["head_camera_type"]]
    args.update(task_name=task_name, task_config=f"demo_{setting}", ckpt_setting="watcher", policy_name="WAtcher",
                eval_mode=True, render_freq=0, save_data=False, eval_video_log=False, eval_video_save_dir=None,
                dual_arm_embodied=True, left_robot_file=str(robot), right_robot_file=str(robot),
                left_embodiment_config=deepcopy(robot_config), right_embodiment_config=deepcopy(robot_config),
                head_camera_h=head["h"], head_camera_w=head["w"])

    def create_scene(engine, config=None):  # CPU physics; render on the policy's GPU
        if config is not None:
            sapien.physx.set_scene_config(config)
        return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(render_device)])
    sapien.Engine.create_scene = create_scene

    task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
    try:
        task.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **args)
        if instruction:
            task.set_instruction(instruction=instruction)
    except BaseException:
        try:  # keep the setup error, even if the partly built scene fails to close
            task.close_env(clear_cache=False)
        except Exception:
            pass
        raise
    return task
