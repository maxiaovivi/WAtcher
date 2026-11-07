"""Open one installed RoboTwin task in an isolated process, without robot I/O."""
from copy import deepcopy
import importlib
import os
from pathlib import Path
import re
import sys


def create_task(root, task_name, setting, seed, instruction, *, render_device="cuda:0"):
    import yaml
    import sapien

    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", task_name):
        raise ValueError("Invalid RoboTwin task name")
    if setting not in ("clean", "randomized"):
        raise ValueError("setting must be clean or randomized")
    if not (root / "envs/_base_task.py").is_file() or not (root / f"envs/{task_name}.py").is_file():
        raise ValueError("Expected an installed RoboTwin source tree and task")
    os.chdir(root)
    for path in (root, root / "policy", root / "description/utils"):
        sys.path.insert(0, str(path))
    config_name = "demo_clean" if setting == "clean" else "demo_randomized"
    def read_yaml(name):
        return yaml.safe_load((root / name).read_text(encoding="utf-8"))
    args = read_yaml(f"task_config/{config_name}.yml")
    if args["embodiment"] != ["aloha-agilex"]:
        raise ValueError("The trained 14D profile requires the official dual aloha-agilex embodiment")
    embeddings = read_yaml("task_config/_embodiment_config.yml")
    robot_root = Path(embeddings["aloha-agilex"]["file_path"])
    if not robot_root.is_absolute():
        robot_root = root / robot_root
    robot_root = robot_root.resolve(strict=True)
    robot_config = yaml.safe_load((robot_root / "config.yml").read_text(encoding="utf-8"))
    if [len(x) for x in robot_config["arm_joints_name"]] != [6, 6]:
        raise ValueError("Expected two six-joint arms")
    cameras = read_yaml("task_config/_camera_config.yml")
    head = cameras[args["camera"]["head_camera_type"]]
    args.update(task_name=task_name, task_config=config_name, ckpt_setting="watcher",
                policy_name="WAtcher", eval_mode=True, render_freq=0, save_data=False,
                eval_video_log=False, eval_video_save_dir=None, dual_arm_embodied=True,
                left_robot_file=str(robot_root), right_robot_file=str(robot_root),
                left_embodiment_config=deepcopy(robot_config), right_embodiment_config=deepcopy(robot_config),
                head_camera_h=head["h"], head_camera_w=head["w"])
    # Bind Vulkan and CUDA to the same process-visible device; retain CPU physics.
    def create_scene(engine, config=None):
        if config is not None:
            sapien.physx.set_scene_config(config)
        return sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(render_device)])
    sapien.Engine.create_scene = create_scene
    module = importlib.import_module(f"envs.{task_name}")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError("A different envs package was already imported; use a fresh process")
    task = getattr(module, task_name)()
    try:
        task.setup_demo(now_ep_num=0, seed=int(seed), is_test=True, **args)
        task.set_instruction(instruction=instruction)
    except BaseException:
        # Upstream may have only partly allocated the scene; retain the original setup error.
        try:
            task.close_env(clear_cache=False)
        except Exception:
            pass
        raise
    return task
