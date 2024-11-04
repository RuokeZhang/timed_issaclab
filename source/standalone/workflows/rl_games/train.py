# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time
import logging
import os
import random
from datetime import datetime

from omni.isaac.lab.app import AppLauncher

# 阶段1：参数解析阶段
start_time_phase1 = time.time()

# 创建主参数解析器
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")

# 添加所有命令行参数，包括 --log_file
parser.add_argument(
    "--log_file",
    type=str,
    default='/ssd/tianshihan/workspace/timed_issaclab/source/standalone/workflows/rl_games/timed.log',
    help="Path to the log file."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# 添加 AppLauncher 的参数
AppLauncher.add_app_launcher_args(parser)

# 解析所有参数，包括之前未解析的参数
args_cli, hydra_args = parser.parse_known_args()

# 如果启用了视频录制，总是启用摄像头
if args_cli.video:
    args_cli.enable_cameras = True

# 清除 sys.argv 以供 Hydra 使用
sys.argv = [sys.argv[0]] + hydra_args

# 阶段1结束，记录时间
end_time_phase1 = time.time()

# 将 args_cli 设置为全局变量，方便在 main 函数中访问
global_args = args_cli

# 阶段2：应用启动阶段
start_time_phase2 = time.time()
# 启动 Omniverse 应用
app_launcher = AppLauncher(global_args)
simulation_app = app_launcher.app
logging.info(f"启动Omniverse应用")
# 阶段2结束，记录时间
end_time_phase2 = time.time()

# 导入剩余的必要模块
import gymnasium as gym
import math

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.assets import retrieve_file_path
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

@hydra_task_config(global_args.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # 在 main 函数内配置日志，确保在 Hydra 初始化后进行
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - PID %(process)d - %(message)s',
        datefmt='%H:%M:%S',
        filename=global_args.log_file,
        filemode='a',  # 使用 'a' 模式以追加日志而不是覆盖
        force=True
    )

    # 记录参数解析的开始时间
    logging.info("参数解析开始")
    start_time_phase1_main = time.time()

    # 阶段3：环境与代理配置阶段
    start_time_phase3 = time.time()
    logging.info("环境与代理配置阶段开始")

    # 覆盖 Hydra 配置中的参数
    env_cfg.scene.num_envs = global_args.num_envs if global_args.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = getattr(global_args, 'device', env_cfg.sim.device) if hasattr(global_args, 'device') else env_cfg.sim.device

    # 如果种子为 -1，则随机生成一个种子
    if global_args.seed == -1:
        global_args.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = global_args.seed if global_args.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        global_args.max_iterations if global_args.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if global_args.checkpoint is not None:
        resume_path = retrieve_file_path(global_args.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        logging.info(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(global_args.sigma) if global_args.sigma is not None else None

    # 多 GPU 训练配置
    if global_args.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # 更新环境配置的设备
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # 设置环境的种子（在多 GPU 配置之后，因为代理的种子可能已更新）
    env_cfg.seed = agent_cfg["params"]["seed"]

    # 指定实验日志目录
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    logging.info(f"[INFO] Logging experiment in directory: {log_root_path}")
    # 指定运行日志目录
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # 设置代理配置中的目录
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # 确保参数目录存在
    params_dir = os.path.join(log_root_path, log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    # 将配置保存到日志目录
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(params_dir, "env.pkl"), env_cfg)
    dump_pickle(os.path.join(params_dir, "agent.pkl"), agent_cfg)

    # 读取代理训练的配置信息
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    # 阶段3结束，记录时间
    end_time_phase3 = time.time()
    logging.info(f"环境与代理配置阶段耗时: {end_time_phase3 - start_time_phase3:.2f} 秒")

    # 阶段4：环境初始化阶段
    start_time_phase4 = time.time()
    logging.info("环境初始化阶段 开始")
    # 创建 Isaac 环境
    env = gym.make(global_args.task, cfg=env_cfg, render_mode="rgb_array" if global_args.video else None)
    # 包装用于视频录制
    if global_args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % global_args.video_interval == 0,
            "video_length": global_args.video_length,
            "disable_logger": True,
        }
        logging.info("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 如果 RL 算法需要，将多代理环境转换为单代理实例
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 包装环境以适应 RL-Games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # 将环境注册到 RL-Games 注册表
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # 将参与者数量设置到代理配置中
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # 阶段4结束，记录时间
    end_time_phase4 = time.time()
    logging.info(f"环境初始化阶段耗时: {end_time_phase4 - start_time_phase4:.2f} 秒")

    # 阶段5：模型训练阶段
    start_time_phase5 = time.time()
    logging.info("模型训练 开始")
    # 从 RL-Games 创建运行器
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # 重置代理和环境
    runner.reset()
    # 训练代理
    if global_args.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})
    # 阶段5结束，记录时间
    end_time_phase5 = time.time()
    logging.info(f"模型训练阶段耗时: {end_time_phase5 - start_time_phase5:.2f} 秒")

    # 阶段6：关闭阶段（环境关闭）
    start_time_phase6_env = time.time()
    logging.info(f"阶段6（环境关闭）开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_phase6_env))}")
    # 关闭模拟器
    env.close()
    # 阶段6结束，记录时间
    end_time_phase6_env = time.time()
    logging.info(f"环境关闭阶段耗时: {end_time_phase6_env - start_time_phase6_env:.2f} 秒")


if __name__ == "__main__":
    # 运行主函数
    main()
    # 阶段6：关闭阶段（应用程序关闭）
    start_time_phase6_app = time.time()
    # 关闭模拟应用
    simulation_app.close()
    # 阶段6结束，记录时间
    end_time_phase6_app = time.time()

    # 由于 logging 已在 main 中配置，因此需要重新配置以记录关闭应用的日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - PID %(process)d - %(message)s',
        filename=global_args.log_file,
        filemode='a',  # 追加模式
        force=True
    )
    logging.info(f"应用程序关闭阶段耗时: {end_time_phase6_app - start_time_phase6_app:.2f} 秒")
