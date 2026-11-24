"""Optional architecture-independent helpers for a masked discrete PPO policy."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    def get_logits(self, obs):
        raise NotImplementedError

    def get_logits_and_value(self, obs):
        raise NotImplementedError

    def get_action_and_value(self, obs, masks, action=None):
        logits, values = self.get_logits_and_value(obs)
        distribution = Categorical(logits=logits.masked_fill(~masks, -torch.inf))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), values

    def forward(self, obs):
        return int(self.act_batch([obs])[0])

    @torch.inference_mode()
    def act_batch(self, observations):
        device = next(self.parameters()).device
        boards = np.stack([obs["observation"] for obs in observations])
        masks = np.stack([obs["action_mask"] for obs in observations])
        boards = torch.as_tensor(boards, dtype=torch.float32, device=device)
        masks = torch.as_tensor(masks, dtype=torch.bool, device=device)
        logits = self.get_logits(boards).masked_fill(~masks, -torch.inf)
        return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1).cpu().numpy()
