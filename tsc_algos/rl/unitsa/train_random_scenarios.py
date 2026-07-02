'''
@Author: WANG Maonan
@Date: 2026-07-01 18:49:01
@Description: UniTSA training with random traffic scenarios on reset
-> python train_random_scenarios.py --junction SouthKorea_Songdo --difficulty easy \
     --num_envs 20 --vec_env subproc --history_len 5 --max_green 50
@LastEditTime: 2026-07-01 22:11:05
@LastEditors: WANG Maonan
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

from tsc_algos.rl.unitsa.unitsa_env.random_scenario_env import make_random_scenario_env
from tsc_algos.rl.unitsa.unitsa_env.make_env import make_env
from tsc_algos.rl.unitsa.model import UniTSAMovementTransformer
from junction_configs import load_junction_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


DEFAULT_PATTERNS = [
    'low_density',
    'high_density',
    'fluctuating_commuter',
    'increasing_demand',
    'random_perturbation',
]

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="WARNING", terminal_log_level="WARNING")


def parse_patterns(patterns: str):
    return [p.strip() for p in patterns.split(',') if p.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTSA 随机场景训练 (PPO)')
    parser.add_argument('--junction', type=str, default='SouthKorea_Songdo',
                        help='路口名称')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'normal'],
                        help='路网难度；会组合为 {difficulty}_{pattern}')
    parser.add_argument('--patterns', type=str, default=','.join(DEFAULT_PATTERNS),
                        help='逗号分隔的交通流模式，默认 5 种 scenario 全部参与 reset 随机采样')
    parser.add_argument('--env_group', type=str, default='',
                        help='输出目录使用的环境组名称；默认 {difficulty}_mixed')
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
                        help='并行环境类型；多环境训练建议 subproc')
    parser.add_argument('--n_steps', type=int, default=256,
                        help='PPO 每个环境一次 rollout 收集的交互步数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='PPO 每次梯度更新的 minibatch 大小')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='PPO 每次 rollout 上重复优化的 epoch 数')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                        help='PPO 策略熵正则系数')
    parser.add_argument('--reward_scale', type=float, default=0.1,
                        help='训练 reward 缩放系数')
    parser.add_argument('--history_len', type=int, default=5,
                        help='UniTSA state/reward 使用的历史帧数')
    parser.add_argument('--max_green', type=int, default=45,
                        help='next_or_not 最大连续绿灯时间；设为 0 则关闭保护')
    parser.add_argument('--features_dim', type=int, default=128,
                        help='policy 接收的特征维度')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='movement token embedding 维度')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Transformer attention heads')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=256,
                        help='Transformer FFN hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Transformer dropout')
    args = parser.parse_args()

    if args.vec_env == 'dummy' and args.num_envs > 1:
        raise ValueError("随机场景多环境训练请使用 --vec_env subproc，避免多个 libsumo 环境共用一个进程。")

    patterns = parse_patterns(args.patterns)
    env_names = [f'{args.difficulty}_{pattern}' for pattern in patterns]
    env_group = args.env_group or f'{args.difficulty}_mixed'

    log_path = path_convert(f'./log/{args.junction}_{env_group}/')
    model_path = path_convert(f'./checkpoints/{args.junction}_{env_group}/')
    tensorboard_path = path_convert(f'./tensorboard/{args.junction}_{env_group}/')
    for p in [log_path, model_path, tensorboard_path]:
        os.makedirs(p, exist_ok=True)

    env_fns = [
        make_random_scenario_env(
            junction=args.junction,
            env_names=env_names,
            use_gui=False,
            log_file=log_path,
            env_index=f'{i}',
            reward_scale=args.reward_scale,
            history_len=args.history_len,
            max_green=args.max_green,
            seed=args.seed + i,
        )
        for i in range(args.num_envs)
    ]
    env = SubprocVecEnv(env_fns) if args.vec_env == 'subproc' else DummyVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)

    # One isolated process per fixed scenario guarantees every evaluation has
    # exactly the same scenario composition. A random reset with five episodes
    # covered all five scenarios only 3.84% of the time.
    eval_env_fns = []
    for eval_index, env_name in enumerate(env_names):
        cfg = load_junction_config(args.junction, env_name)
        eval_env_fns.append(make_env(
            tls_id=cfg['tls_id'],
            num_seconds=cfg['num_seconds'],
            num_phases=cfg['num_phases'],
            sumo_cfg=cfg['sumo_cfg'],
            net_file=cfg['net_file'],
            use_gui=False,
            log_file=log_path,
            env_index=f'eval_{eval_index}',
            reward_scale=args.reward_scale,
            history_len=args.history_len,
            max_green=args.max_green,
        ))
    eval_env = (
        SubprocVecEnv(eval_env_fns)
        if len(eval_env_fns) > 1
        else DummyVecEnv(eval_env_fns)
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        training=False,
        gamma=0.99,
    )

    ppo_params = {
        'learning_rate': 1e-4,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': args.ent_coef,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=model_path,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=log_path,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=len(env_names),
        deterministic=True,
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Random reset env_names: {env_names}")
    logger.info(f"PPO params: {ppo_params}")

    policy_kwargs = dict(
        features_extractor_class=UniTSAMovementTransformer,
        features_extractor_kwargs=dict(
            features_dim=args.features_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ),
        activation_fn=torch.nn.GELU,
        share_features_extractor=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        **ppo_params,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_path,
        device=device,
        seed=args.seed,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name=f'{args.junction}_{env_group}',
        callback=callback_list,
    )

    model.save(f'{model_path}/{args.junction}_{env_group}.zip')
    env.save(f'{model_path}/vec_normalize.pkl')
    print('随机场景训练结束, 达到最大步数.')

    env.close()
    eval_env.close()
