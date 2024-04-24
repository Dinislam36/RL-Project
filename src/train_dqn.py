import random

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torchvision.transforms import Resize

from src.environment import BOWAPEnv
from src.dqn import DQNAgent


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


@hydra.main(version_base=None, config_path="../configs", config_name="train_dqn.yaml")
def main(cfg: DictConfig) -> None:
    learning_rate = 3e-4
    discount_factor = 0.5
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    replay_buffer_size = 50000
    batch_size = 32
    num_episodes = 10

    size = tuple([int(x) for x in cfg.state_dim])
    resize = Resize(size)

    env = BOWAPEnv()
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    agent = DQNAgent(state_space, action_space, learning_rate, discount_factor, epsilon, replay_buffer_size, batch_size, device)

    for episode in range(num_episodes):
        state = env.reset()
        state = (
            resize(torch.tensor(state).permute((2, 0, 1)).unsqueeze(0))
            .float()
            .to(device)
        )  # Preprocess state (e.g., resize, convert to grayscale)
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            print(action)
            next_state, reward, done, info = env.step(action)
            next_state = (
                resize(torch.tensor(next_state).permute((2, 0, 1)).unsqueeze(0))
                .float()
                .to(device)
            )
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn(device)

            state = next_state
            total_reward += reward

            # Update epsilon for exploration decay
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode: {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")


if __name__ == "__main__":
    main()