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
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
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

    def select_action(self, state):
        probabilities = F.softmax(self.policy_network(state), dim=1)
        action_probs = torch.distributions.Categorical(probabilities)
        
        action = action_probs.sample().item()
        action_tensor = torch.tensor([action])
        action_tensor = action_tensor.cuda()
        log_prob = action_probs.log_prob(action_tensor)
        
        return action, log_prob


    def store_experience(self, state, action, reward, next_state, log_prob):
        self.memory.push(state, action, reward, next_state)
        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)

    def update_policy_gradient(self):
        R = 0
        policy_losses = []
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.tensor(returns, requires_grad=False)
        returns_tensor = returns_tensor.cuda()
        normalized_returns = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-5)

        for log_prob, R in zip(self.saved_log_probs, normalized_returns):
            policy_losses.append(-log_prob * R)

        # Eğer policy_losses boşsa, işlemleri gerçekleştirmeyelim.
        if len(policy_losses) == 0:
            print("Warning: policy_losses list is empty!")
            return

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.saved_log_probs.clear()

    def end_of_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Epsilon Value: {self.epsilon:.4f}")
        self.update_policy_gradient()