from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import optim
from torchvision import transforms
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image

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
        value_loss += F.mse_loss(value.squeeze().to(device), reward.squeeze().to(device))
        # Total loss
    return action_loss / len(actions), value_loss / len(values)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    num_actions = cfg.num_actions
    env = BOWAPEnv(random=cfg.random, survive_time=cfg.survive_time,
                   terminate_on_death=cfg.terminate_on_death,
                   num_actions=num_actions, num_bullets_spawn=cfg.num_bullets_spawn,
                   bullets_speed=cfg.bullets_speed,
                   survive_reward=cfg.survive_reward,
                   death_reward=cfg.death_reward,
                   hold_shift=cfg.hold_shift)
    size = tuple([int(x) for x in cfg.state_dim])
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    learning_rate = cfg.lr

    gamma = cfg.gamma

    # Initialize models and optimizer
    device = torch.device(cfg.device)
    actor = Actor(cfg.state_dim, env.action_space.n).to(device)
    critic = Critic(cfg.state_dim).to(device).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate / 10)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    #actor_scheduler = CustomScheduler(actor_optimizer)
    #critic_scheduler = CustomScheduler(critic_optimizer)

    max_frames = 0
    total_deaths = 1e8
    eps = cfg.eps_initial
    # Training loop
    for episode in range(cfg.num_episodes):
        state = env.reset()
        # Preprocess state
        state = transform(Image.fromarray(state)).unsqueeze(0).to(device)
        done = False
        rewards = []
        actions = []
        values = []

        for i in range(3000):
            # Forward pass through Actor to get action probabilities
            if True: # i % cfg.frame_step == 0:
                action_probs = actor(state)
                action_distribution = Categorical(action_probs)
                # Sample action
                if np.random.rand() < eps:
                    action = np.random.randint(0, env.action_space.n-1)
                else:
                    action = action_distribution.sample().item()
            # Take action in the environment
            next_state, reward, done, info = env.step(action)
            if i % cfg.frame_step == 0 or info['dead'] or done:
                rewards.append(reward)

                # Preprocess next state

                # Critic prediction for current state and action
                Q_value = critic(state)
                actions.append((action_distribution.log_prob(torch.Tensor([action]).to(device)), action))
                values.append(Q_value)
                state = transform(Image.fromarray(next_state)).unsqueeze(0).to(device)
            if done:
                break
        eps = max(cfg.eps_final, eps * cfg.eps_decay)
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

        if cfg.terminate_on_death and info['frames'] >= max_frames:
            max_frames = info['frames']
            torch.save(actor.state_dict(), "../weights/actor.pt")
            torch.save(critic.state_dict(), "../weights/critic.pt")
        if not cfg.terminate_on_death and info['total_deaths'] <= total_deaths:
            total_deaths = info['total_deaths']
            torch.save(actor.state_dict(), "../weights/actor.pt")
            torch.save(critic.state_dict(), "../weights/critic.pt")

if __name__ == "__main__":
    main()
