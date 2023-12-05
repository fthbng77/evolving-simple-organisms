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

    #log_prob aksiyonun politika ağı tarafından ne kadar tercih edildiğini ifade eder.
    def select_action(self, state):
        sample = random.random()

        if state.size(1) != 8:
            state = state[:, :8]

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if torch.cuda.is_available() and not state.is_cuda:
            state = state.cuda()
        
        print("State shape:", state.shape)
        print("state size: ", state.size())

        with torch.no_grad():
            policy, _ = self.policy_network(state)
            probabilities = torch.distributions.Categorical(policy)
            device = policy.device
            if sample > self.epsilon:
                action = probabilities.sample().item()
            else:
                action = random.randrange(self.action_size)

            action_tensor = torch.tensor([action], device=device)
            log_prob = probabilities.log_prob(action_tensor).item()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
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
            advantage = advantage.detach()
            policy_loss = -log_prob * advantage
            value_loss = F.mse_loss(torch.tensor([value]).cuda(), torch.tensor([R]).cuda())
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

        if not policy_losses or not value_losses:
            print("No losses to update policy.")
            return
        
        #Burada actor-critic yapısı var politika kayıpları ve deer kayıpları ayrı ayrı hesaplanıyor.
        if len(policy_losses) > 1:
            policy_loss = torch.cat(policy_losses).sum().clone().detach().requires_grad_(True)
        elif policy_losses:
            policy_loss = policy_losses[0].clone().detach().requires_grad_(True)

        if len(value_losses) > 1:
            value_loss = torch.cat(value_losses).sum().clone().detach().requires_grad_(True)
        elif value_losses:
            value_loss = value_losses[0].clone().detach().requires_grad_(True)

        if policy_losses:
            policy_loss.backward()
        if value_losses:
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
