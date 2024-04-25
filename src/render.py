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
import time

np.random.seed(0)
torch.manual_seed(0)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    env = BOWAPEnv(random=False,
                   survive_time=45,
                   terminate_on_death=cfg.terminate_on_death,
                   num_actions=cfg.num_actions,
                   num_bullets_spawn=cfg.num_bullets_spawn,
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

    import torch

    device = torch.device(cfg.device)
    actor = Actor(cfg.state_dim, env.action_space.n).to(device)
    critic = Critic(cfg.state_dim).to(device).to(device)
    actor.load_state_dict(torch.load("../weights/actor.pt"))
    actor.eval()
    critic.load_state_dict(torch.load("../weights/critic.pt"))
    critic.eval()

    start = time.time()
    state = env.reset()
    # Preprocess state
    imgs = [state]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    state = transform(Image.fromarray(state)).unsqueeze(0).to(device)
    done = False
    rewards = []

    while not done:
        # Forward pass through Actor to get action probabilities
        with torch.no_grad():
            action_probs = actor(state)
        action = Categorical(action_probs).sample().item()  # Sample action

        # Take action in the environment
        next_state, reward, done, info = env.step(action)
        imgs.append(next_state)
        rewards.append(reward)

        # Preprocess next state
        state = transform(Image.fromarray(next_state)).unsqueeze(0).to(device)
    end = time.time()
    print(f"Time elapsed: {end - start:.4f}")
    print(f"Total reward: {sum(rewards)}")
    print(f"Info: {info}")

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Change this to your desired codec

    video = cv2.VideoWriter("result.avi", fourcc, 60.0, (576, 672))

    for image in imgs:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
