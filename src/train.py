from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import optim
from torchvision.transforms import Resize

from src.models import Actor, Critic
from src.environment import BOWAPEnv


np.random.seed(0)
torch.manual_seed(0)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    env = BOWAPEnv()
    resize = Resize(cfg.state_dim)

    learning_rate = cfg.lr
    gamma = cfg.gamma

    # Initialize models and optimizer
    device = torch.device(cfg.device)
    actor = Actor(cfg.state_dim, env.action_space.n).to(device)
    critic = Critic(cfg.state_dim).to(device).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        # Preprocess state
        state = (
            resize(torch.tensor(state).permute((2, 0, 1)).unsqueeze(0))
            .float()
            .to(device)
        )
        done = False
        rewards = []
        while not done:
            # Forward pass through Actor to get action probabilities
            action_probs = actor(state)
            action = torch.multinomial(action_probs, 1).item()  # Sample action

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # Preprocess next state
            next_state = (
                resize(torch.tensor(next_state).permute((2, 0, 1)).unsqueeze(0))
                .float()
                .to(device)
            )

            # Critic prediction for current state and action
            Q_value = critic(state)

            # Bellman equation for target Q-value
            if not done:
                target_Q = reward + gamma * critic(next_state)
            else:
                target_Q = torch.tensor(reward).view(1,1).to(device)

            # Critic loss (MSE between predicted and target Q-value)
            critic_loss = torch.nn.functional.mse_loss(Q_value, target_Q.detach())
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Advantage calculation (estimated future reward - critic prediction)
            advantage = target_Q - Q_value.detach()  # Detach for stopping gradients

            actor_loss = -torch.log(action_probs[0, action]) * advantage.item()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update state for next iteration
            state = next_state
        # Print episode stats (optional)
        print(f"Episode: {episode}, Total Reward: {sum(rewards)}")
        Path("../weights").mkdir(exist_ok=True)
        torch.save(actor.state_dict(), "../weights/actor.pt")
        torch.save(critic.state_dict(), "../weights/critic.pt")


if __name__ == "__main__":
    main()
