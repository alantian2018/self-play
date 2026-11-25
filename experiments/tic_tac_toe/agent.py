import torch
import torch.nn as nn

from selfplay.policy import Policy, layer_init


class Agent(Policy):
    policy_version = 2

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

        cells = torch.arange(9).reshape(3, 3)
        permutations = torch.stack([
            torch.rot90(cells.flip(1) if reflected else cells, k, (0, 1)).flatten()
            for reflected in (False, True) for k in range(4)
        ])
        # Each transformed cell points to its original observation/action index.
        # Nonpersistent buffers preserve compatibility with existing weight files.
        self.register_buffer("symmetry_indices", permutations, persistent=False)
        self.register_buffer("inverse_indices", permutations.argsort(dim=1), persistent=False)

    def symmetric_features(self, obs):
        boards = obs.reshape(-1, 9, 2)[:, self.symmetry_indices]
        return self.encoder(boards.reshape(-1, 3, 3, 2).permute(0, 3, 1, 2))

    def symmetric_logits(self, features):
        logits = self.actor(features).reshape(-1, 8, 9)
        # Undo each action permutation before combining the eight predictions.
        aligned = logits.gather(2, self.inverse_indices[None].expand(logits.shape[0], -1, -1))
        return aligned.mean(dim=1)

    def get_logits(self, obs):
        return self.symmetric_logits(self.symmetric_features(obs))

    def get_logits_and_value(self, obs):
        features = self.symmetric_features(obs)
        values = self.critic(features).reshape(-1, 8).mean(dim=1)
        return self.symmetric_logits(features), values
