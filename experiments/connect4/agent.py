import torch
import torch.nn as nn

from selfplay.policy import Policy, layer_init


class Agent(Policy):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(2, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        self.actor = layer_init(nn.Linear(32 * 6 * 7, 7), std=0.01)
        self.critic = layer_init(nn.Linear(32 * 6 * 7, 1), std=1.0)

    def mirrored_features(self, obs):
        # NHWC: dimension 2 is columns. Evaluate both views in one batch.
        boards = torch.cat((obs, obs.flip(2)), dim=0)
        return self.encoder(boards.permute(0, 3, 1, 2))

    def get_logits(self, obs):
        original, mirrored = self.actor(self.mirrored_features(obs)).chunk(2)
        return (original + mirrored.flip(-1)) / 2

    def get_logits_and_value(self, obs):
        features = self.mirrored_features(obs)
        original, mirrored = self.actor(features).chunk(2)
        values = self.critic(features).squeeze(-1).reshape(2, -1).mean(0)
        return (original + mirrored.flip(-1)) / 2, values
