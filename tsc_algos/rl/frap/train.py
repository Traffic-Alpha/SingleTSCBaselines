'''
@Author: WANG Maonan
@Description: FRAP 训练脚本
LastEditTime: 2026-04-14 20:40:18
'''
import os
import argparse
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from tsc_algos.output_utils import generate_output_paths
from tsc_algos.rl.utils.experiment_tracker import TrainingSummaryCallback
from .env.make_env import make_env
from .models.mlp import MLPJunctionModel
from junction_configs import load_junction_config

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="INFO")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FRAP 训练')
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
    parser.add_argument('--reward_scale', type=float, default=0.01,
                        help='训练 reward 缩放系数')
    args = parser.parse_args()

    cfg = load_junction_config(args.junction, args.env_name)

    log_path = path_convert('./log/')
    model_path = path_convert('./models/')
    tensorboard_path = path_convert('./tensorboard/')
    for p in [log_path, model_path, tensorboard_path]:
        os.makedirs(p, exist_ok=True)

    # 生成 SUMO 输出文件路径
    trip_info, fcd_output = generate_output_paths(args.junction, args.env_name, "frap")
    eval_trip_info, eval_fcd_output = generate_output_paths(args.junction, args.env_name, "frap_eval")

    # #########
    # Init Env
    # #########
    params = {
        'tls_id': cfg['tls_id'],
        'num_seconds': cfg['num_seconds'],
        'num_phases': cfg['num_phases'],
        'sumo_cfg': cfg['sumo_cfg'],
        'net_file': cfg['net_file'],
        'use_gui': False,
        'log_file': log_path,
        'reward_scale': args.reward_scale,
        'trip_info': trip_info,
        'fcd_output': fcd_output,
    }
    env_fns = [make_env(env_index=f'{i}', **params) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns) if args.vec_env == 'subproc' else DummyVecEnv(env_fns)

    eval_params = dict(params)
    eval_params.update({
        'log_file': path_convert('./eval_log/'),
        'trip_info': eval_trip_info,
        'fcd_output': eval_fcd_output,
    })
    eval_env = DummyVecEnv([make_env(env_index='eval', **eval_params)])

    # #########
    # Callback
    # #########
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=model_path,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=path_convert('./eval/'),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
    )
    summary_callback = TrainingSummaryCallback(
        summary_path=path_convert('../training_summary.csv'),
        method='frap',
        junction=args.junction,
        env_name=args.env_name,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        eval_npz_path=path_convert('./eval/evaluations.npz'),
        extra={
            'trainer': 'DQN',
            'num_envs': args.num_envs,
            'vec_env': args.vec_env,
            'reward_scale': args.reward_scale,
        },
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback, summary_callback])

    # #########
    # Training
    # #########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    policy_kwargs = dict(
        features_extractor_class=MLPJunctionModel,
        features_extractor_kwargs=dict(
            features_dim=128,
            hidden_dims=[256, 256],
        ),
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        gamma=0.99,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_path,
        device=device,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.junction, callback=callback_list)

    # #################
    # 保存 model 和 env
    # #################
    model.save(f'{model_path}/last_rl_model.zip')
    print('训练结束, 达到最大步数.')

    env.close()
    eval_env.close()
