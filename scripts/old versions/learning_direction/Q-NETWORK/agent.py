import torch
import torch.nn.functional as F
import torch.optim as optim
from qnetwork import QNetwork, ReplayMemory
import random
from tensorboardX import SummaryWriter

class Agent:
    def __init__(self, input_dim, output_dim, hidden_dim=32, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.998, epsilon_min=0.05):
        self.qnetwork = QNetwork(input_dim, output_dim, hidden_dim).cuda()
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = output_dim
        self.memory = ReplayMemory(55000)
        self.writer = SummaryWriter(log_dir='./logs')
        self.global_step = 0

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            with torch.no_grad():
                q_values = self.qnetwork(state)
                return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def update(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        states, actions, rewards, next_states = zip(*transitions)

        states = torch.cat(states).cuda()
        actions = torch.tensor(actions, dtype=torch.long).cuda().view(-1, 1)
        rewards = torch.tensor(rewards).cuda().view(-1, 1)
        next_states = torch.cat(next_states).cuda()

        current_q_values = self.qnetwork(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.qnetwork(next_states)
            max_next_q_values, _ = next_q_values.max(dim=1, keepdim=True)
            target_q_values = rewards + (self.gamma * max_next_q_values)

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('Loss', loss.item(), self.global_step)
        self.global_step += 1

    def end_of_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Epsilon Value: {self.epsilon:.4f}")
