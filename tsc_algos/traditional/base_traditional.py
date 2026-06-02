'''
@Author: WANG Maonan
@Description: 传统算法基类，运行在 TSCInfoWrapper 之上
'''
from typing import Dict, Any, List


class BaseTraditionalAgent:
    """传统 TSC 算法基类

    子类需要实现 choose_action 方法。
    运行在 TSCInfoWrapper 之上，直接读取字典格式的特征。
    """

    def choose_action(
        self,
        lane_dynamic_features_seq: List[Dict[str, Any]],
        static_lane_features: Dict[str, Any],
    ) -> int:
        """根据特征序列选择动作

        Args:
            lane_dynamic_features_seq: 上一个决策间隔内所有子步的特征序列
                seq[-1] 为最后一帧（当前状态），seq[0] 为最早一帧
            static_lane_features: lane 静态特征

        Returns:
            action: 动作（整数）
        """
        raise NotImplementedError

    def _get_lane_dynamic_features_seq(self, obs):
        """读取 TSCInfoWrapper 返回的 lane 动态特征序列。"""
        return obs['lane_dynamic_features_seq']

    def run(self, env, num_episodes: int = 1):
        """运行算法

        Args:
            env: TSCInfoWrapper 环境
            num_episodes: 运行轮数
        """
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_steps = 0

            while not done:
                action = self.choose_action(
                    lane_dynamic_features_seq=self._get_lane_dynamic_features_seq(obs),
                    static_lane_features=env.static_lane_features,
                )
                obs, reward, truncated, done, info = env.step(action)
                total_steps += 1

            print(f"Episode {ep+1}: steps={total_steps}")

        env.close()
