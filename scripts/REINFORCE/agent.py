import torch
import torch.nn.functional as F
import torch.optim as optim
from policynetwork import PolicyNetwork, ReplayMemory
import random
from tensorboardX import SummaryWriter

#torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim=128, learning_rate=0.0001, gamma=0.9, epsilon=0.8, epsilon_decay=1, epsilon_min=0.05):
        torch.autograd.set_detect_anomaly(True)
        self.policy_network = PolicyNetwork(output_dim, input_dim, hidden_dim).cuda()
        self.actor_optimizer = optim.Adam(self.policy_network.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.policy_network.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = output_dim
        self.memory = ReplayMemory(200000)
        self.writer = SummaryWriter(log_dir='./logs')
        self.global_step = 0
        self.rewards = []
        self.saved_log_probs = []
        self.values = []
        self.death_count = 0
        self.device = torch.device('cuda')

    #log_prob aksiyonun politika ağı tarafından ne kadar tercih edildiğini ifade eder.
    def select_action(self, state):
        sample = random.random()

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).cuda()
        elif not state.is_cuda:
            state = state.cuda()

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

        return action, log_prob


    def store_experience(self, state, action, reward, next_state, log_prob):
        self.memory.push(state, action, reward, next_state)
        self.rewards.append(reward)
        self.saved_log_probs.append(log_prob)

        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        else:
            state_tensor = state.to(self.device)

        value = self.policy_network.actor(state_tensor)
        self.values.append(value.mean().item())



    def update_policy_gradient(self):
        if len(self.rewards) == 0:
            print("No rewards to update policy.")
            return
        
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
        self.death_count += 1
        if self.death_count > 100:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Epsilon Value: {self.epsilon:.4f}")
        self.update_policy_gradient()
