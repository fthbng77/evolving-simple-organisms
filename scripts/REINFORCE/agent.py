import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from policynetwork import PolicyNetwork
from tensorboardX import SummaryWriter

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim=64, learning_rate=0.0003,
                 gamma=0.99, entropy_coefficient=0.005, entropy_decay=0.99,
                 entropy_minimum=0.0005, device=None, log_dir='./logs'):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(input_dim, output_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.policy_network.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.policy_network.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coefficient = entropy_coefficient
        self.entropy_decay = entropy_decay
        self.entropy_minimum = entropy_minimum
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None
        self.global_step = 0
        self.rewards = []
        self.saved_log_probs = []
        self.values = []
        self.entropies = []

    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            self.policy_network.eval()
            with torch.no_grad():
                probabilities, _ = self.policy_network(state)
            return torch.argmax(probabilities.squeeze(0)).item(), None, None, None

        self.policy_network.train()
        probabilities, value = self.policy_network(state)
        action_probs = torch.distributions.Categorical(probs=probabilities.squeeze(0))
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        return action.item(), log_prob, value.squeeze(0).squeeze(0), action_probs.entropy()

    def store_step(self, reward, log_prob, value, entropy):
        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

    def finish_episode(self):
        if not self.rewards:
            return None

        returns = torch.zeros(len(self.rewards), dtype=torch.float32, device=self.device)
        # Fill returns backwards without converting network outputs to Python numbers.
        discounted_return = 0.0
        for index in range(len(self.rewards) - 1, -1, -1):
            discounted_return = self.rewards[index] + self.gamma * discounted_return
            returns[index] = discounted_return

        values = torch.stack(self.values)
        log_probs = torch.stack(self.saved_log_probs)
        entropies = torch.stack(self.entropies)
        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropies.mean()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        metrics = {
            'return': returns[0].item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropies.mean().item(),
        }
        self.rewards.clear()
        self.saved_log_probs.clear()
        self.values.clear()
        self.entropies.clear()
        self.entropy_coefficient = max(
            self.entropy_minimum,
            self.entropy_coefficient * self.entropy_decay,
        )
        metrics['entropy_coefficient'] = self.entropy_coefficient
        return metrics

    def save_policy(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + '.tmp')
        torch.save({
            'model_state_dict': self.policy_network.state_dict(),
            'global_step': self.global_step,
            'metadata': metadata,
        }, temporary_path)
        temporary_path.replace(path)

    def load_policy(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint.get('metadata', {})

    @staticmethod
    def checkpoint_metadata(path):
        return torch.load(path, map_location='cpu').get('metadata', {})
