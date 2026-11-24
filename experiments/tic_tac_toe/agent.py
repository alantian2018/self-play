import torch.nn as nn

from selfplay.policy import Policy, layer_init


class Agent(Policy):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
       #     layer_init(nn.Conv2d(2, 16, 2, padding=1)), nn.ReLU(),
       #     layer_init(nn.Conv2d(16, 16, 2, padding=1)), nn.ReLU(),
       #     layer_init(nn.Conv2d(16, 32, 2, padding=1)), nn.ReLU(),
                     
            nn.Flatten(),
            nn.Linear(18, 128),
            nn.ReLU()
        )
     
        self.actor = layer_init(nn.Linear(128, 9), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def get_logits(self, obs):
        features = self.encoder(obs.permute(0, 3, 1, 2))
        return self.actor(features)

    def get_logits_and_value(self, obs):
        features = self.encoder(obs.permute(0, 3, 1, 2))
        return self.actor(features), self.critic(features).squeeze(-1)
