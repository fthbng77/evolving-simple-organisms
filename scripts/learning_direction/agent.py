import torch
import torch.nn.functional as F
import torch.optim as optim
from qnetwork import QNetwork
import random

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim=32, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.qnetwork = QNetwork(input_dim, output_dim, hidden_dim).cuda()
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = output_dim

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.qnetwork(state)
                return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        current_q_values = self.qnetwork(state)
        with torch.no_grad():
            next_q_values = self.qnetwork(next_state)
            target_q_value = reward + self.gamma * torch.max(next_q_values)
        
        loss = self.loss_fn(current_q_values[0][action], target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)