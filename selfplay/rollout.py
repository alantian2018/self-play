import numpy as np


def copy_observation(obs):
    return {key: value.copy() for key, value in obs.items()}


class Rollout:
    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.terminated = []
        self.truncated = []

    def add_step(self, obs, action, reward, next_obs, terminated, truncated):
        self.obs.append(copy_observation(obs))
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(copy_observation(next_obs))
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def get_episode(self):
        """Stack each field along the time axis, keeping boards and masks."""
        def stack_obs(observations):
            if not observations:
                return {}
            return {
                key: np.stack([obs[key] for obs in observations])
                for key in observations[0]
            }

        return {
            "obs": stack_obs(self.obs),
            "action": np.array(self.action, dtype=np.int64),
            "reward": np.array(self.reward, dtype=np.float32),
            "next_obs": stack_obs(self.next_obs),
            "terminated": np.array(self.terminated, dtype=bool),
            "truncated": np.array(self.truncated, dtype=bool),
        }
