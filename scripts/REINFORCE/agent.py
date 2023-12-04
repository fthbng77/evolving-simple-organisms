import torch
import torch.nn.functional as F
import torch.optim as optim
from policynetwork import PolicyNetwork, ReplayMemory
import random
from tensorboardX import SummaryWriter

#torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim=32, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        torch.autograd.set_detect_anomaly(True)
        self.policy_network = PolicyNetwork(input_dim, output_dim, hidden_dim).cuda()
        self.actor_optimizer = optim.Adam(self.policy_network.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.policy_network.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = output_dim
        self.memory = ReplayMemory(75000)
        self.writer = SummaryWriter(log_dir='./logs')
        self.global_step = 0
        self.rewards = []
        self.saved_log_probs = []
        self.values = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).cuda()
        probabilities, value = self.policy_network(state)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        action = action.item()

        self.values.append(value.item())
        return action, log_prob


    def store_experience(self, state, action, reward, next_state, log_prob):
        self.memory.push(state, action, reward, next_state)
        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)

    def update_policy_gradient(self):
        R = 0
        returns = []
        policy_losses = []
        value_losses = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).cuda()
        values = torch.tensor(self.values).cuda()
        advantages = returns - values

        for log_prob, value, R, advantage in zip(self.saved_log_probs, self.values, returns, advantages):
            advantage = advantage.detach()  # avantajı sabitle
            policy_loss = -log_prob * advantage
            value_loss = F.mse_loss(torch.tensor([value]).cuda(), torch.tensor([R]).cuda())
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        policy_loss = torch.cat(policy_losses).sum()
        value_loss = torch.cat(value_losses).sum()

        policy_loss.backward()
        value_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.rewards.clear()
        self.saved_log_probs.clear()
        self.values.clear()

    def end_of_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Epsilon Value: {self.epsilon:.4f}")
        self.update_policy_gradient()