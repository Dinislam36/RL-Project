import random
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class DQN(nn.Module):
    def __init__(self, in_channels, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7744, 256)  # Adjust based on output of conv layers
        self.fc2 = nn.Linear(256, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return  torch.cat(state, dim=0).float(), torch.LongTensor(action), torch.FloatTensor(reward), torch.cat(next_state, dim=0).float(), torch.FloatTensor(done)

    def __len__(self):
        return len(self.buffer)
    

class DQNAgent(object):
    def __init__(self, state_space, action_space, learning_rate, discount_factor, epsilon, replay_buffer_size, batch_size, device):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DQN(3, action_space).to(device)
        self.target_net = DQN(3, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Initialize target net with policy net weights
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.policy_net(state)
        return q_values.argmax(1).item()

    def learn(self, device):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples for training

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Double DQN for stability (optional)
        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states).detach()  # Detach for stopping gradients in double DQN
        expected_q_values = rewards.to(device).unsqueeze(1) + self.discount_factor * torch.max(next_q_values, dim=1)[0].unsqueeze(1)
        

        # Huber loss for robustness to outliers
        loss = nn.functional.smooth_l1_loss(q_values, expected_q_values)
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically (e.g., every few learning steps)
        self.update_target_network()

    def update_target_network(self, tau=0.125):  # Update target network with a polyak update rule (optional)
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
