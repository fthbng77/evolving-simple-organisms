import torch
import torch.nn as nn
import torch.optim as optim
import random 
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, learning_rate=1e-3, dropout_prob=0.2):
        super(QNetwork, self).__init__()
        
        self.dropout_prob = dropout_prob


        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            #nn.Linear(hidden_dim, output_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.fc(x)

    def update(self, state, target_q_values):
        predicted_q_values = self(state)
        loss = self.criterion(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
