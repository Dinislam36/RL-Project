import random

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.environment import BOWAPEnv
from src.dqn import DQNAgent


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


@hydra.main(version_base=None, config_path="../configs", config_name="train_dqn.yaml")
def main(cfg: DictConfig) -> None:
    learning_rate = cfg.learning_rate
    discount_factor = 0.5
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.01
    epsilon_decay = cfg.epsilon_decay
    replay_buffer_size = 50000
    batch_size = 32
    num_episodes = cfg.num_episodes

    size = tuple([int(x) for x in cfg.state_dim])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    env = BOWAPEnv(random=cfg.random, survive_time=cfg.survive_time,
                   terminate_on_death=cfg.terminate_on_death,
                   num_actions=cfg.num_actions, num_bullets_spawn=cfg.num_bullets_spawn,
                   bullets_speed=cfg.bullets_speed,
                   survive_reward=cfg.survive_reward,
                   death_reward=cfg.death_reward)
    state_space = env.observation_space.shape
    action_space = env.action_space.n
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    agent = DQNAgent(state_space, action_space, learning_rate, discount_factor, epsilon, replay_buffer_size, batch_size, device)

    for episode in range(num_episodes):
        state = env.reset()
        # Preprocess state (e.g., resize, convert to grayscale)
        state = transform(Image.fromarray(state)).unsqueeze(0).to(device)
        done = False
        total_reward = 0
        i = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = transform(Image.fromarray(next_state)).unsqueeze(0).to(device)
            if i % cfg.frame_step == 0 or info['dead'] or done:
                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.learn(device)
            state = next_state
            total_reward += reward
            i += 1
        # Update epsilon for exploration decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        agent.update_epsilon(epsilon)

        torch.save(agent.policy_net.state_dict(), "../weights/dqn.pth")
        print(f"Episode: {episode}, Total Reward: {total_reward:.4f},"
              f" Epsilon: {epsilon:.4f}")
        print(f"Episode info:\t{info}")


if __name__ == "__main__":
    main()