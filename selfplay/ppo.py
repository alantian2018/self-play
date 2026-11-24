"""PPO updates adapted from CleanRL; independent of the game environment."""
import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def make_batch(agent, rollouts, args):
    """Compute old log probabilities and GAE before changing the policy."""
    device = next(agent.parameters()).device
    batches = []
    for rollout in rollouts:
        episode = rollout.get_episode()
        obs = torch.as_tensor(episode["obs"]["observation"], dtype=torch.float32, device=device)
        masks = torch.as_tensor(episode["obs"]["action_mask"], dtype=torch.bool, device=device)
        actions = torch.as_tensor(episode["action"], device=device)
        rewards = torch.as_tensor(episode["reward"], device=device)
        _, logprobs, _, values = agent.get_action_and_value(obs, masks, actions)

        # Each rollout is a complete game, so its final bootstrap value is zero.
        advantages = torch.zeros_like(rewards)
        next_value = advantage = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + args.gamma * next_value - values[t]
            advantage = delta + args.gamma * args.gae_lambda * advantage
            advantages[t] = advantage
            next_value = values[t]
        returns = advantages + values
        batches.append((obs, masks, actions, logprobs, advantages, returns))

    # Concatenate only after GAE so advantages never cross game boundaries.
    return tuple(torch.cat(column) for column in zip(*batches))


def update(agent, optimizer, rollouts, args):
    obs, masks, actions, old_logprobs, advantages, returns = make_batch(agent, rollouts, args)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    metrics = []

    for _ in range(args.update_epochs):
        indices = torch.randperm(len(actions), device=actions.device)
        for minibatch in indices.split(args.minibatch_size):
            _, logprobs, entropy, values = agent.get_action_and_value(
                obs[minibatch], masks[minibatch], actions[minibatch],
            )
            logratio = logprobs - old_logprobs[minibatch]
            ratio = logratio.exp()
            unclipped = -advantages[minibatch] * ratio
            clipped = -advantages[minibatch] * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)
            policy_loss = torch.maximum(unclipped, clipped).mean()
            value_loss = 0.5 * (values - returns[minibatch]).square().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            metrics.append([
                policy_loss.item(), value_loss.item(), entropy.mean().item(),
                ((ratio - 1) - logratio).mean().item(),
                ((ratio - 1).abs() > args.clip_coef).float().mean().item(),
                len(minibatch),
            ])

    metrics = np.array(metrics)
    return dict(zip(
        ("losses/policy", "losses/value", "losses/entropy", "losses/approx_kl", "losses/clip_fraction"),
        np.average(metrics[:, :-1], axis=0, weights=metrics[:, -1]),
    ))
