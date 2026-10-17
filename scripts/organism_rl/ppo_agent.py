from pathlib import Path

import torch
import torch.nn.functional as functional
import torch.optim as optim
from tensorboardX import SummaryWriter

from policynetwork import PolicyNetwork


class PPOAgent:
    """Shared-policy PPO agent with one rollout per organism."""

    def __init__(
        self, input_dim, output_dim, hidden_dim=64, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, update_epochs=4,
        entropy_coefficient=0.01, value_coefficient=0.5, device=None, log_dir=None,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.trajectories = {}
        self.global_step = 0

    def select_action(self, state, deterministic=False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            self.policy_network.eval()
            with torch.no_grad():
                probabilities, _ = self.policy_network(state_tensor)
            return torch.argmax(probabilities.squeeze(0)).item()
        self.policy_network.train()
        probabilities, value = self.policy_network(state_tensor)
        distribution = torch.distributions.Categorical(probs=probabilities.squeeze(0))
        action = distribution.sample()
        return action.item(), distribution.log_prob(action).detach(), value.squeeze().detach()

    def select_actions(self, states, deterministic=False):
        """Sample a batch of actions in one network pass for population simulations."""
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self.policy_network.train(not deterministic)
        with torch.no_grad() if deterministic else torch.enable_grad():
            probabilities, values = self.policy_network(state_tensor)
        if deterministic:
            actions = torch.argmax(probabilities, dim=-1)
            return [(action.item(), None, value.detach()) for action, value in zip(actions, values.squeeze(-1))]
        distribution = torch.distributions.Categorical(probs=probabilities)
        actions = distribution.sample()
        return [
            (action.item(), log_prob.detach(), value.detach())
            for action, log_prob, value in zip(actions, distribution.log_prob(actions), values.squeeze(-1))
        ]

    def store_step(self, trajectory_id, state, action, old_log_prob, value, reward, terminal):
        trajectory = self.trajectories.setdefault(
            trajectory_id,
            {'states': [], 'actions': [], 'log_probs': [], 'values': [], 'rewards': [], 'terminals': []},
        )
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['log_probs'].append(old_log_prob)
        trajectory['values'].append(value)
        trajectory['rewards'].append(reward)
        trajectory['terminals'].append(terminal)

    def finish_rollouts(self, trajectory_ids):
        trajectories = [self.trajectories.pop(index) for index in trajectory_ids if index in self.trajectories]
        trajectories = [trajectory for trajectory in trajectories if trajectory['rewards']]
        if not trajectories:
            return None

        states, actions, old_log_probs, returns, advantages = [], [], [], [], []
        for trajectory in trajectories:
            rewards = trajectory['rewards']
            values = [value.item() if isinstance(value, torch.Tensor) else value for value in trajectory['values']]
            gae = 0.0
            trajectory_advantages = [0.0] * len(rewards)
            for index in range(len(rewards) - 1, -1, -1):
                next_value = 0.0 if index == len(rewards) - 1 else values[index + 1]
                nonterminal = 1.0 - float(trajectory['terminals'][index])
                delta = rewards[index] + self.gamma * next_value * nonterminal - values[index]
                gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
                trajectory_advantages[index] = gae
            states.extend(trajectory['states'])
            actions.extend(trajectory['actions'])
            old_log_probs.extend(trajectory['log_probs'])
            advantages.extend(trajectory_advantages)
            returns.extend(advantage + value for advantage, value in zip(trajectory_advantages, values))

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        actor_loss = critic_loss = entropy = 0.0
        for _ in range(self.update_epochs):
            probabilities, values = self.policy_network(states)
            distribution = torch.distributions.Categorical(probs=probabilities)
            ratios = torch.exp(distribution.log_prob(actions) - old_log_probs)
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss_tensor = -torch.minimum(unclipped, clipped).mean()
            critic_loss_tensor = functional.smooth_l1_loss(values.squeeze(-1), returns)
            entropy_tensor = distribution.entropy().mean()
            loss = actor_loss_tensor + self.value_coefficient * critic_loss_tensor - self.entropy_coefficient * entropy_tensor
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
            self.optimizer.step()
            actor_loss, critic_loss, entropy = actor_loss_tensor.item(), critic_loss_tensor.item(), entropy_tensor.item()
        return {'actor_loss': actor_loss, 'critic_loss': critic_loss, 'entropy': entropy, 'return': returns.mean().item()}

    def save_policy(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + '.tmp')
        torch.save({'model_state_dict': self.policy_network.state_dict(), 'global_step': self.global_step, 'metadata': metadata}, temporary_path)
        temporary_path.replace(path)

    def load_policy(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint.get('metadata', {})
