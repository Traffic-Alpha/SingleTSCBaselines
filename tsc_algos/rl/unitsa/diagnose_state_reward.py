'''Inspect whether UniTSA state and last-frame reward track traffic congestion.'''
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from junction_configs import load_junction_config
from tsc_algos.rl.unitsa.unitsa_env.make_env import make_env
from tsc_algos.rl.unitsa.unitsa_env.reward_funcs import (
    movement_avg_accumulated_waiting_time,
)
from tsc_algos.rl.unitsa.unitsa_env.state_funcs import (
    MAX_ACCUMULATED_WAITING_TIME,
    MAX_JAM_VEHICLES,
    MAX_SPEED,
    MAX_VEHICLES,
)


DEFAULT_PATTERNS = [
    'low_density',
    'high_density',
    'fluctuating_commuter',
    'increasing_demand',
    'random_perturbation',
]

path_convert = get_abs_path(__file__)
logger.remove()
set_logger(path_convert('./'), file_log_level='WARNING', terminal_log_level='WARNING')


def parse_patterns(patterns: str):
    return [pattern.strip() for pattern in patterns.split(',') if pattern.strip()]


def weighted_mean(values, weights):
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        return 0.0
    return float(np.sum(np.asarray(values) * np.asarray(weights)) / total_weight)


def choose_action(action_mode: str, rng: np.random.Generator):
    if action_mode == 'always_switch':
        return 0
    if action_mode == 'always_keep':
        return 1
    return int(rng.integers(0, 2))


def collect_scenario(args, env_name: str):
    cfg = load_junction_config(args.junction, env_name)
    env = make_env(
        tls_id=cfg['tls_id'],
        num_seconds=cfg['num_seconds'],
        num_phases=cfg['num_phases'],
        sumo_cfg=cfg['sumo_cfg'],
        net_file=cfg['net_file'],
        history_len=args.history_len,
        max_green=args.max_green,
        reward_scale=args.reward_scale,
    )()
    rng = np.random.default_rng(args.action_seed)
    records = []
    try:
        state, _info = env.reset()
        terminated = False
        truncated = False
        decision = 0
        while not (terminated or truncated):
            action = choose_action(args.action_mode, rng)
            state, reward, terminated, truncated, info = env.step(action)
            decision += 1

            movement_frame = env.env.tls_dynamic_features_seq[-1]
            cell_counts = np.asarray([m['vehicle_count'] for m in movement_frame])
            detector_counts = np.asarray([
                len(m['last_step_vehicle_id_list']) for m in movement_frame
            ])
            accumulated_waits = np.asarray([
                m['avg_accumulated_waiting_time'] for m in movement_frame
            ])
            speeds = np.asarray([m['mean_speed'] for m in movement_frame])
            occupancies = np.asarray([m['occupancy'] for m in movement_frame])
            jams = np.asarray([m['jam_length_vehicle'] for m in movement_frame])
            raw_waiting_cost = movement_avg_accumulated_waiting_time(movement_frame)

            last_state = state[-1]
            state_cell_counts = np.expm1(
                last_state[:, 0] * np.log1p(MAX_VEHICLES)
            )
            state_speeds = last_state[:, 2] * MAX_SPEED
            state_jams = last_state[:, 3] * MAX_JAM_VEHICLES
            state_waits = np.expm1(
                last_state[:, 6] * np.log1p(MAX_ACCUMULATED_WAITING_TIME)
            )

            records.append({
                'scenario': env_name,
                'decision': decision,
                'sim_time': int(info['step_time']),
                'decision_interval_seconds': int(info['decision_interval_seconds']),
                'action': int(info['applied_action']),
                'reward': float(reward),
                'raw_waiting_cost': raw_waiting_cost,
                'reward_abs_error': abs(
                    float(reward) + args.reward_scale * raw_waiting_cost
                ),
                'cell_vehicle_count': float(cell_counts.sum()),
                'detector_vehicle_count': float(detector_counts.sum()),
                'state_cell_vehicle_count': float(state_cell_counts.sum()),
                'weighted_speed': weighted_mean(speeds, cell_counts),
                'state_weighted_speed': weighted_mean(state_speeds, state_cell_counts),
                'mean_occupancy': float(occupancies.mean()),
                'state_mean_occupancy': float(last_state[:, 1].mean()),
                'jam_vehicle_count': float(jams.sum()),
                'state_jam_vehicle_count': float(state_jams.sum()),
                'state_reconstructed_waiting_cost': weighted_mean(state_waits, cell_counts),
                'vehicle_count_clip_fraction': float(np.mean(last_state[:, 0] >= 1.0)),
                'speed_clip_fraction': float(np.mean(last_state[:, 2] >= 1.0)),
                'jam_clip_fraction': float(np.mean(last_state[:, 3] >= 1.0)),
                'waiting_clip_fraction': float(np.mean(last_state[:, 6] >= 1.0)),
                'zero_history_frames': int(np.sum(np.abs(state).sum(axis=(1, 2)) == 0)),
            })
    finally:
        env.close()
    return records


def build_summary(steps: pd.DataFrame):
    rows = []
    for scenario, frame in steps.groupby('scenario', sort=False):
        rows.append({
            'scenario': scenario,
            'decisions': len(frame),
            'reward_mean': frame['reward'].mean(),
            'reward_min': frame['reward'].min(),
            'waiting_mean': frame['raw_waiting_cost'].mean(),
            'waiting_p95': frame['raw_waiting_cost'].quantile(0.95),
            'waiting_max': frame['raw_waiting_cost'].max(),
            'cell_vehicles_mean': frame['cell_vehicle_count'].mean(),
            'detector_vehicles_mean': frame['detector_vehicle_count'].mean(),
            'speed_mean': frame['weighted_speed'].mean(),
            'occupancy_mean': frame['mean_occupancy'].mean(),
            'jam_mean': frame['jam_vehicle_count'].mean(),
            'reward_error_max': frame['reward_abs_error'].max(),
            'state_wait_mae': np.mean(np.abs(
                frame['state_reconstructed_waiting_cost'] - frame['raw_waiting_cost']
            )),
            'vehicle_clip_mean': frame['vehicle_count_clip_fraction'].mean(),
            'jam_clip_mean': frame['jam_clip_fraction'].mean(),
            'waiting_clip_mean': frame['waiting_clip_fraction'].mean(),
            'corr_wait_jam': frame['raw_waiting_cost'].corr(
                frame['jam_vehicle_count'], method='spearman'
            ),
            'corr_wait_occupancy': frame['raw_waiting_cost'].corr(
                frame['mean_occupancy'], method='spearman'
            ),
            'corr_wait_inverse_speed': frame['raw_waiting_cost'].corr(
                -frame['weighted_speed'], method='spearman'
            ),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junction', default='SouthKorea_Songdo')
    parser.add_argument('--difficulty', default='easy', choices=['easy', 'normal'])
    parser.add_argument('--patterns', default=','.join(DEFAULT_PATTERNS))
    parser.add_argument('--action_mode', default='always_switch',
                        choices=['always_switch', 'always_keep', 'random'])
    parser.add_argument('--action_seed', type=int, default=1)
    parser.add_argument('--history_len', type=int, default=5)
    parser.add_argument('--max_green', type=int, default=50)
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--output_root', default='')
    args = parser.parse_args()

    env_names = [
        f'{args.difficulty}_{pattern}' for pattern in parse_patterns(args.patterns)
    ]
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else PROJECT_ROOT / 'results' / 'unitsa_state_reward_diagnostics'
    )
    run_dir = output_root / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)

    records = []
    for env_name in env_names:
        print(f'Collecting {env_name} with {args.action_mode} ...')
        records.extend(collect_scenario(args, env_name))

    steps = pd.DataFrame(records)
    summary = build_summary(steps)
    steps.to_csv(run_dir / 'steps.csv', index=False, float_format='%.6f')
    summary.to_csv(run_dir / 'summary.csv', index=False, float_format='%.6f')

    print('\nState/reward diagnostic summary:')
    print(summary.to_string(index=False, float_format=lambda value: f'{value:.3f}'))
    print(f'\nDetailed output: {run_dir}')


if __name__ == '__main__':
    main()
