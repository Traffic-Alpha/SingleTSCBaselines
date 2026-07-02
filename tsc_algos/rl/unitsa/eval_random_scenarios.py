'''
@Author: WANG Maonan
@Description: Evaluate the final UniTSA mixed-scenario model on every fixed scenario.

Example:
    python eval_random_scenarios.py \
        --junction SouthKorea_Songdo --difficulty easy \
        --episodes_per_scenario 1 --history_len 5 --max_green 50
'''
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from junction_configs import load_junction_config
from stable_baselines3 import PPO
from tsc_algos.rl.unitsa.model import UniTSAMovementTransformer
from tsc_algos.rl.unitsa.unitsa_env.make_env import make_env


path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level='WARNING', terminal_log_level='WARNING')

DEFAULT_PATTERNS = [
    'low_density',
    'high_density',
    'fluctuating_commuter',
    'increasing_demand',
    'random_perturbation',
]
CHECKPOINT_RE = re.compile(r'^rl_model_(\d+)_steps\.zip$')


def parse_patterns(patterns: str):
    parsed = [pattern.strip() for pattern in patterns.split(',') if pattern.strip()]
    if not parsed:
        raise ValueError('At least one traffic pattern is required.')
    return parsed


def resolve_model_path(
    checkpoint_dir: Path,
    junction: str,
    env_group: str,
    explicit_model_path: str = '',
) -> Path:
    """Select the final model, or fall back to the numerically latest checkpoint.

    best_model.zip is deliberately never considered.
    """
    if explicit_model_path:
        model_path = Path(explicit_model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'Model does not exist: {model_path}')
        return model_path

    final_model = checkpoint_dir / f'{junction}_{env_group}.zip'
    if final_model.is_file():
        return final_model

    numbered_checkpoints = []
    for model_path in checkpoint_dir.glob('rl_model_*_steps.zip'):
        match = CHECKPOINT_RE.match(model_path.name)
        if match:
            numbered_checkpoints.append((int(match.group(1)), model_path))

    if not numbered_checkpoints:
        raise FileNotFoundError(
            f'No final model or numbered checkpoint found in {checkpoint_dir}'
        )
    return max(numbered_checkpoints, key=lambda item: item[0])[1]


def summarize_tripinfo(trip_info_path: Path):
    """Read traffic-level metrics from one SUMO tripinfo output."""
    if not trip_info_path.is_file():
        return {
            'num_vehicles': 0,
            'travel_time_mean': np.nan,
            'travel_time_max': np.nan,
            'waiting_time_mean': np.nan,
            'waiting_time_p95': np.nan,
            'waiting_time_max': np.nan,
            'time_loss_mean': np.nan,
        }

    tripinfos = ET.parse(trip_info_path).getroot().findall('tripinfo')
    if not tripinfos:
        return {
            'num_vehicles': 0,
            'travel_time_mean': np.nan,
            'travel_time_max': np.nan,
            'waiting_time_mean': np.nan,
            'waiting_time_p95': np.nan,
            'waiting_time_max': np.nan,
            'time_loss_mean': np.nan,
        }

    duration = np.asarray([float(item.attrib['duration']) for item in tripinfos])
    waiting = np.asarray([float(item.attrib['waitingTime']) for item in tripinfos])
    time_loss = np.asarray([float(item.attrib['timeLoss']) for item in tripinfos])
    return {
        'num_vehicles': len(tripinfos),
        'travel_time_mean': float(duration.mean()),
        'travel_time_max': float(duration.max()),
        'waiting_time_mean': float(waiting.mean()),
        'waiting_time_p95': float(np.percentile(waiting, 95)),
        'waiting_time_max': float(waiting.max()),
        'time_loss_mean': float(time_loss.mean()),
    }


def evaluate_episode(
    model: PPO,
    junction: str,
    env_name: str,
    episode_index: int,
    output_dir: Path,
    history_len: int,
    max_green: int,
    reward_scale: float,
    use_gui: bool,
):
    cfg = load_junction_config(junction, env_name)
    trip_info_path = output_dir / f'{env_name}_episode_{episode_index:03d}_tripinfo.xml'
    params = {
        'tls_id': cfg['tls_id'],
        'num_seconds': cfg['num_seconds'],
        'num_phases': cfg['num_phases'],
        'sumo_cfg': cfg['sumo_cfg'],
        'net_file': cfg['net_file'],
        'use_gui': use_gui,
        'history_len': history_len,
        'max_green': max_green,
        'reward_scale': reward_scale,
        'trip_info': str(trip_info_path),
    }
    env = make_env(env_index=f'{env_name}_{episode_index}', **params)()

    episode_reward = 0.0
    episode_steps = 0
    simulated_seconds = 0
    forced_switches = 0
    applied_switches = 0
    switch_probabilities = []
    try:
        obs, _info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            obs_tensor, _vectorized = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                action_probs = (
                    model.policy.get_distribution(obs_tensor)
                    .distribution.probs[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            switch_probabilities.append(float(action_probs[0]))
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1
            simulated_seconds += int(info.get('decision_interval_seconds', 0))
            forced_switches += int(info.get('max_green_forced', False))
            applied_switches += int(info.get('applied_action') == 0)
    finally:
        env.close()

    mean_decision_waiting_cost = (
        -episode_reward / (reward_scale * episode_steps)
        if reward_scale > 0 and episode_steps > 0
        else np.nan
    )
    result = {
        'scenario': env_name,
        'episode': episode_index,
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'simulated_seconds': simulated_seconds,
        'mean_decision_waiting_cost': mean_decision_waiting_cost,
        'applied_switches': applied_switches,
        'max_green_forced': forced_switches,
        'switch_probability_mean': float(np.mean(switch_probabilities)),
        'switch_probability_min': float(np.min(switch_probabilities)),
        'switch_probability_max': float(np.max(switch_probabilities)),
        'tripinfo_path': str(trip_info_path),
    }
    result.update(summarize_tripinfo(trip_info_path))
    return result


def build_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        'episode_reward': ['mean', 'median', 'std', 'min', 'max'],
        'episode_steps': ['mean'],
        'simulated_seconds': ['mean'],
        'mean_decision_waiting_cost': ['mean', 'median', 'max'],
        'num_vehicles': ['mean'],
        'travel_time_mean': ['mean'],
        'travel_time_max': ['mean', 'max'],
        'waiting_time_mean': ['mean'],
        'waiting_time_p95': ['mean'],
        'waiting_time_max': ['mean', 'max'],
        'time_loss_mean': ['mean'],
        'applied_switches': ['mean'],
        'max_green_forced': ['mean'],
        'switch_probability_mean': ['mean', 'min', 'max'],
    }
    summary = episodes.groupby('scenario', sort=False).agg(metrics)
    summary.columns = [f'{column}_{stat}' for column, stat in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the final UniTSA model on each fixed traffic scenario.'
    )
    parser.add_argument('--junction', type=str, default='SouthKorea_Songdo')
    parser.add_argument('--difficulty', type=str, default='easy', choices=['easy', 'normal'])
    parser.add_argument('--patterns', type=str, default=','.join(DEFAULT_PATTERNS))
    parser.add_argument('--env_group', type=str, default='',
                        help='Mixed training group; defaults to {difficulty}_mixed.')
    parser.add_argument('--episodes_per_scenario', type=int, default=1)
    parser.add_argument('--history_len', type=int, default=5)
    parser.add_argument('--max_green', type=int, default=45,
                        help='Must match the mixed-scenario training setting.')
    parser.add_argument('--reward_scale', type=float, default=0.1,
                        help='Used only to reproduce logged reward and unscale its reporting.')
    parser.add_argument('--model_path', type=str, default='',
                        help='Optional explicit model. Default: final model, then latest numbered checkpoint.')
    parser.add_argument('--output_root', type=str, default='',
                        help='Output root; each invocation creates a timestamped subdirectory.')
    parser.add_argument('--gui', action='store_true', default=False)
    args = parser.parse_args()

    if args.episodes_per_scenario <= 0:
        parser.error('--episodes_per_scenario must be positive.')
    if args.reward_scale <= 0:
        parser.error('--reward_scale must be positive.')

    patterns = parse_patterns(args.patterns)
    env_names = [f'{args.difficulty}_{pattern}' for pattern in patterns]
    env_group = args.env_group or f'{args.difficulty}_mixed'
    checkpoint_dir = Path(__file__).resolve().parent / 'checkpoints' / f'{args.junction}_{env_group}'
    model_path = resolve_model_path(
        checkpoint_dir=checkpoint_dir,
        junction=args.junction,
        env_group=env_group,
        explicit_model_path=args.model_path,
    )

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else PROJECT_ROOT / 'results' / 'unitsa_random_scenario_eval' / f'{args.junction}_{env_group}'
    )
    run_dir = output_root / f'{model_path.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = UniTSAMovementTransformer  # Ensure SB3 can resolve the custom extractor class.
    model = PPO.load(str(model_path), device=device)

    print(f'Model: {model_path}')
    print(f'Scenarios: {env_names}')
    print(f'Output: {run_dir}')

    records = []
    for env_name in env_names:
        for episode_index in range(1, args.episodes_per_scenario + 1):
            result = evaluate_episode(
                model=model,
                junction=args.junction,
                env_name=env_name,
                episode_index=episode_index,
                output_dir=run_dir,
                history_len=args.history_len,
                max_green=args.max_green,
                reward_scale=args.reward_scale,
                use_gui=args.gui,
            )
            records.append(result)
            print(
                f'[{env_name} #{episode_index}] '
                f'reward={result["episode_reward"]:.2f}, '
                f'P(switch)={result["switch_probability_mean"]:.3f}, '
                f'waiting={result["waiting_time_mean"]:.2f}, '
                f'waiting_p95={result["waiting_time_p95"]:.2f}, '
                f'waiting_max={result["waiting_time_max"]:.2f}'
            )

    episodes = pd.DataFrame(records)
    summary = build_summary(episodes)
    episodes_path = run_dir / 'episodes.csv'
    summary_path = run_dir / 'summary.csv'
    episodes.to_csv(episodes_path, index=False, float_format='%.6f')
    summary.to_csv(summary_path, index=False, float_format='%.6f')

    display_columns = [
        'scenario',
        'episode_reward_mean',
        'mean_decision_waiting_cost_mean',
        'waiting_time_mean_mean',
        'waiting_time_p95_mean',
        'waiting_time_max_max',
        'travel_time_mean_mean',
        'switch_probability_mean_mean',
    ]
    print('\nScenario summary:')
    print(summary[display_columns].to_string(index=False, float_format=lambda value: f'{value:.2f}'))
    print(f'\nEpisode results: {episodes_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
