from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import optim
from torchvision.transforms import Resize
import torch.nn.functional as F
from torch.distributions import Categorical

from src.models import Actor, Critic
from src.environment import BOWAPEnv
from src.scheduler import CustomScheduler


np.random.seed(0)
torch.manual_seed(0)


def calc_loss(rewards, actions, values, gamma, device):
    returns = []
    # Calculate discounted rewards
    discounted_reward = 0
    for reward in rewards[::-1]:
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    action_loss = 0
    value_loss = 0
    # Calc loss
    for (logprob, action), reward, value in zip(actions, returns, values):
        # Compair reward to Critic prediction
        gain = reward - value.item()
        # How good action was
        action_loss += -logprob * gain

        # How good V-value was predicted
        value_loss += F.smooth_l1_loss(value.to(device), reward.to(device))
        # Total loss
    return action_loss, value_loss


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    num_actions = cfg.num_actions
    env = BOWAPEnv(random=True, survive_time=cfg.survive_time,
                   terminate_on_death=cfg.terminate_on_death,
                   num_actions=num_actions, num_bullets_spawn=cfg.num_bullets_spawn,
                   bullets_speed=cfg.bullets_speed)
    size = tuple([int(x) for x in cfg.state_dim])
    resize = Resize(size)

    learning_rate = cfg.lr

    gamma = cfg.gamma

    # Initialize models and optimizer
    device = torch.device(cfg.device)
    actor = Actor(cfg.state_dim, env.action_space.n).to(device)
    critic = Critic(cfg.state_dim).to(device).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    #actor_scheduler = CustomScheduler(actor_optimizer)
    #critic_scheduler = CustomScheduler(critic_optimizer)

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
        actions = []
        values = []
        for i in range(3000):
            # Forward pass through Actor to get action probabilities
            action_probs = actor(state)
            action_distribution = Categorical(action_probs)
            # Sample action
            action = action_distribution.sample()
            # Take action in the environment
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)

            # Preprocess next state
            next_state = (
                resize(torch.tensor(next_state).permute((2, 0, 1)).unsqueeze(0))
                .float()
                .to(device)
            )
            # Critic prediction for current state and action
            Q_value = critic(state)
            actions.append((action_distribution.log_prob(action), action))
            values.append(Q_value)

            if done:
                break

        actor_loss, critic_loss = calc_loss(rewards, actions, values, gamma, device)
        # Actor optimizer step
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # Critic optimizer step
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Print episode stats (optional)
        print(f"Episode: {episode}, Total Reward: {sum(rewards):.4f},"
              f" Actor loss: {actor_loss.item():.4f}, critic_loss: {critic_loss.item():.4f}")
        print(f"Episode info:\t{info}")
        Path("../weights").mkdir(exist_ok=True)
        torch.save(actor.state_dict(), "../weights/actor.pt")
        torch.save(critic.state_dict(), "../weights/critic.pt")


if __name__ == "__main__":
    main()
