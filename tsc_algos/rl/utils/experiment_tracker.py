'''
@Author: WANG Maonan
@Description: Lightweight experiment tracking utilities for RL training
'''
import csv
import os
import time
from datetime import datetime
from statistics import mean
from typing import Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


SUMMARY_FIELDNAMES = [
    'timestamp',
    'method',
    'junction',
    'env_name',
    'seed',
    'target_timesteps',
    'trained_timesteps',
    'elapsed_seconds',
    'episode_reward_mean',
    'episode_length_mean',
    'trainer',
    'num_envs',
    'vec_env',
    'reward_scale',
    'eval_last_reward_mean',
    'eval_best_reward_mean',
    'eval_last_length_mean',
]


class TrainingSummaryCallback(BaseCallback):
    """Append one row to a CSV summary when training finishes."""

    def __init__(
        self,
        summary_path: str,
        method: str,
        junction: str,
        env_name: str,
        total_timesteps: int,
        seed: int,
        eval_npz_path: Optional[str] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.summary_path = summary_path
        self.method = method
        self.junction = junction
        self.env_name = env_name
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.eval_npz_path = eval_npz_path
        self.extra = extra or {}
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        ep_info = list(getattr(self.model, 'ep_info_buffer', []))
        ep_rewards = [episode['r'] for episode in ep_info if 'r' in episode]
        ep_lengths = [episode['l'] for episode in ep_info if 'l' in episode]

        eval_stats = self._load_eval_stats()
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': self.method,
            'junction': self.junction,
            'env_name': self.env_name,
            'seed': self.seed,
            'target_timesteps': self.total_timesteps,
            'trained_timesteps': self.num_timesteps,
            'elapsed_seconds': round(elapsed, 3),
            'episode_reward_mean': round(mean(ep_rewards), 6) if ep_rewards else '',
            'episode_length_mean': round(mean(ep_lengths), 6) if ep_lengths else '',
            **eval_stats,
            **self.extra,
        }
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        existing_rows = []
        existing_fieldnames = []
        if os.path.exists(self.summary_path):
            with open(self.summary_path, newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)

        fieldnames = list(existing_fieldnames) if existing_fieldnames else list(SUMMARY_FIELDNAMES)
        for fieldname in SUMMARY_FIELDNAMES:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
        for fieldname in row.keys():
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)

        for fieldname in fieldnames:
            row.setdefault(fieldname, '')
        for existing_row in existing_rows:
            for fieldname in fieldnames:
                existing_row.setdefault(fieldname, '')

        with open(self.summary_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)

    def _load_eval_stats(self) -> Dict[str, object]:
        if not self.eval_npz_path or not os.path.exists(self.eval_npz_path):
            return {}

        data = np.load(self.eval_npz_path)
        results = data['results'] if 'results' in data.files else None
        lengths = data['ep_lengths'] if 'ep_lengths' in data.files else None
        if results is None or len(results) == 0:
            return {}

        reward_means = results.mean(axis=1)
        stats = {
            'eval_last_reward_mean': round(float(reward_means[-1]), 6),
            'eval_best_reward_mean': round(float(reward_means.max()), 6),
        }
        if lengths is not None and len(lengths) > 0:
            stats['eval_last_length_mean'] = round(float(lengths.mean(axis=1)[-1]), 6)
        return stats
