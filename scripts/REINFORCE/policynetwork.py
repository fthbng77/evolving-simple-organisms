import torch
import torch.nn as nn
import torch.optim as optim
import random

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, dropout_prob=0.2):
        super(PolicyNetwork, self).__init__()
        self.dropout_prob = dropout_prob

        # Aktör Ağı (Politika Ağı)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # Eleştirmen Ağı (Değerlendirme Ağı)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden_dim, 1)  # Durumun değerini tahmin etmek için tek bir çıktı
        )

    def forward(self, x):
        policy = self.actor(x)  # Politika ağından eylem olasılıkları
        value = self.critic(x)  # Değer ağından durumun değeri
        return policy, value

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