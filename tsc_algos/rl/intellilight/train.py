'''
@Author: WANG Maonan
@Description: IntelliLight 训练脚本（= PressLight 的 DQN，reward 换成累计等待时间 + VecNormalize）
-> python train.py --junction Beijing_Beihuan --env_name normal_increasing_demand \
    --num_envs 20 --reward_scale 0.1 --vec_env subproc --history_len 5
'''
import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from tsc_algos.output_utils import generate_output_paths
from tsc_algos.rl.intellilight.intellilight_env.make_env import make_env
from tsc_algos.rl.intellilight.model import IntelliLightMovementModel
from junction_configs import load_junction_config

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="WARNING", terminal_log_level="WARNING")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IntelliLight 训练 (DQN)')
    parser.add_argument('--junction', type=str, default='Beijing_Beihuan',
                        help='路口名称')
    parser.add_argument('--env_name', type=str, default='easy_low_density',
                        help='环境名称，如 easy_low_density')
    parser.add_argument('--total_timesteps', type=int, default=300000,
                        help='训练总步数')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子')
    parser.add_argument('--checkpoint_freq', type=int, default=10000,
                        help='checkpoint 保存频率')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='deterministic evaluation 频率')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='并行训练环境数量')
    parser.add_argument('--vec_env', type=str, default='dummy', choices=['dummy', 'subproc'],
                        help='并行环境类型；沙箱内建议 dummy，本机训练可用 subproc')
    parser.add_argument('--episode_steps', type=int, default=150,
                        help='单次仿真大约包含的 RL 交互步数，用于派生 DQN 更新节奏')
    parser.add_argument('--reward_scale', type=float, default=0.01,
                        help='训练 reward 缩放系数')
    parser.add_argument('--history_len', type=int, default=4,
                        help='IntelliLight state/reward 使用的历史帧数')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    log_path = path_convert(f'./log/{args.junction}_{args.env_name}/')
    model_path = path_convert(f'./checkpoints/{args.junction}_{args.env_name}/')
    tensorboard_path = path_convert(f'./tensorboard/{args.junction}_{args.env_name}/')
    for p in [log_path, model_path, tensorboard_path]:
        os.makedirs(p, exist_ok=True)

    eval_trip_info, eval_fcd_output = generate_output_paths(args.junction, args.env_name, "intellilight_eval")

    params = {
        'tls_id': cfg['tls_id'],
        'num_seconds': cfg['num_seconds'],
        'num_phases': cfg['num_phases'],
        'sumo_cfg': cfg['sumo_cfg'],
        'net_file': cfg['net_file'],
        'use_gui': False,
        'log_file': log_path,
        'reward_scale': args.reward_scale,
        'history_len': args.history_len,
    }
    env_fns = [make_env(env_index=f'{i}', **params) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns) if args.vec_env == 'subproc' else DummyVecEnv(env_fns)
    # 仅对 reward 做归一化 (norm_obs=False)：累计等待 reward 尺度大且会随 starvation 增长，
    # 归一化稳住学习。norm_obs 必须 False：obs 已归一化到 [0,1] 且 padding 靠全 0 识别。
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)

    eval_params = dict(params)
    eval_params.update({
        'log_file': log_path,
        'trip_info': eval_trip_info,
        'fcd_output': eval_fcd_output,
    })
    # eval_env 也包成 VecNormalize（norm_reward=False, training=False）：EvalCallback 会
    # sync_envs_normalization(训练env, eval_env)，要求两者同为 VecEnvWrapper；eval reward
    # 仍为原始值。独立的 eval.py 不走 EvalCallback，无需 VecNormalize。
    eval_env = DummyVecEnv([make_env(env_index='eval', **eval_params)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False,
                            training=False, gamma=0.99)

    dqn_params = {
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'learning_starts': max(args.episode_steps * 8, 1000),
        'batch_size': 64,
        'train_freq': 4,
        'gradient_steps': max(args.num_envs, 1),
        'gamma': 0.99,
        'exploration_fraction': 0.2,
        'exploration_final_eps': 0.05,
        'target_update_interval': max(args.episode_steps * 5, 500),
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=model_path,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"DQN params: {dqn_params}")

    policy_kwargs = dict(
        features_extractor_class=IntelliLightMovementModel,
        features_extractor_kwargs=dict(
            features_dim=128,
            hidden_dims=[128, 128],
        ),
    )

    model = DQN(
        "MlpPolicy",
        env,
        **dqn_params,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_path,
        device=device,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.junction, callback=callback_list)

    model.save(f'{model_path}/{args.junction}_{args.env_name}.zip')
    env.save(f'{model_path}/vec_normalize.pkl')  # 仅用于复现/续训，eval.py 不需要
    print('训练结束, 达到最大步数.')

    env.close()
    eval_env.close()
