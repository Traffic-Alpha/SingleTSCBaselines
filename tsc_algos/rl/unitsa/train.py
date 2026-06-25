'''
@Author: WANG Maonan
@Description: UniTSA 训练脚本（PPO 版本）
-> python train.py --junction Beijing_Beihuan --env_name normal_increasing_demand \
     --num_envs 20 --reward_scale 0.1 --vec_env subproc --history_len 5
@LastEditTime: 2026-06-26 00:24:40
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
from tsc_algos.rl.unitsa.unitsa_env.make_env import make_env
from tsc_algos.rl.unitsa.model import UniTSAMovementTransformer
from junction_configs import load_junction_config

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level="WARNING", terminal_log_level="WARNING")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UniTSA 训练 (PPO)')
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
    parser.add_argument('--n_steps', type=int, default=256,
                        help='PPO 每个环境一次 rollout 收集的交互步数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='PPO 每次梯度更新的 minibatch 大小')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='PPO 每次 rollout 上重复优化的 epoch 数')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                        help='PPO 策略熵正则系数（需配合 share_features_extractor=False）。'
                             '设 0 时确定性策略会间歇塌缩(eval 抖动)；过大(如 0.01)又可能'
                             '压住本任务较弱的 advantage 信号。0.005 作为熵下限既防塌缩又不压学习。')
    parser.add_argument('--reward_scale', type=float, default=0.1,
                        help='训练 reward 缩放系数')
    parser.add_argument('--history_len', type=int, default=5,
                        help='UniTSA state/reward 使用的历史帧数')
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

    cfg = load_junction_config(args.junction, args.env_name)

    log_path = path_convert(f'./log/{args.junction}_{args.env_name}/')
    model_path = path_convert(f'./checkpoints/{args.junction}_{args.env_name}/')
    tensorboard_path = path_convert(f'./tensorboard/{args.junction}_{args.env_name}/')
    for p in [log_path, model_path, tensorboard_path]:
        os.makedirs(p, exist_ok=True)

    eval_trip_info, eval_fcd_output = generate_output_paths(args.junction, args.env_name, "unitsa_eval")

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
    # 仅对 reward 做归一化 (norm_obs=False)，稳定 PPO 的 value/advantage 尺度。
    # 注意 norm_obs 必须为 False：obs 已在 state_funcs 归一化到 [0,1]，且 Transformer
    # 用 `obs.abs().sum(-1)==0` 识别 padding，归一化 obs 会破坏该 mask。
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.99)

    eval_params = dict(params)
    eval_params.update({
        'log_file': log_path,
        'trip_info': eval_trip_info,
        'fcd_output': eval_fcd_output,
    })
    # eval_env 也必须包成 VecNormalize：当训练 env 是 VecNormalize 时，EvalCallback 会
    # 调用 sync_envs_normalization(训练env, eval_env)，要求两者同为 VecEnvWrapper，否则
    # 第一次 eval 直接 AssertionError 崩溃。这里 norm_reward=False + training=False，使
    # eval reward 仍为原始值、且不更新自身统计量（只接收训练 env 同步过来的统计量）。
    # 独立的 eval.py 评估脚本不走 EvalCallback，因此无需 VecNormalize。
    eval_env = DummyVecEnv([make_env(env_index='eval', **eval_params)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False,
                            training=False, gamma=0.99)

    ppo_params = {
        'learning_rate': 3e-4,
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
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
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
        # actor/critic 各自独立的主干。共享时 value_loss 较大, 会把共享主干的特征
        # 塑造成"预测状态价值"(与动作无关), 冲掉区分动作的信息, 导致 policy 学不动
        # (熵贴在均匀分布)。这是 PPO 收敛的决定性配置。
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
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.junction, callback=callback_list)

    model.save(f'{model_path}/{args.junction}_{args.env_name}.zip')
    env.save(f'{model_path}/vec_normalize.pkl')  # 仅用于复现/续训，eval.py 不需要
    print('训练结束, 达到最大步数.')

    env.close()
    eval_env.close()
