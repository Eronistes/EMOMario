import os
import json
import time
import warnings
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from pathlib import Path
import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as T
import gym_super_mario_bros
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape, antialias=True), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        self.online = self.build_cnn(c, h, w, output_dim)
        self.target = self.build_cnn(c, h, w, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

    def build_cnn(self, c, h, w, output_dim):
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        return cnn

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.save_dir = Path(save_dir)
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): print(f'Using:  {self.device}')
        self.net = MarioNet(state_dim, action_dim).to(device=self.device)
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        self.batch_size = 32
        self.gamma = 0.9
        self.lr = 0.00025
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.sync_every = 10000
        self.burnin = 10000
        self.learn_every = 4

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_q = self.net(state, model="online")[torch.arange(0, self.batch_size), action]
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()
        if self.curr_step < self.burnin or self.curr_step % self.learn_every != 0:
            return None, None
        state, next_state, action, reward, done = self.recall()
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(reward, next_state, done)
        loss = self.update_q_online(td_estimate, td_target)
        return (td_estimate.mean().item(), loss)

    def save(self, step):
        save_path = self.save_dir / f"mario_net_{step}.chkpt"
        torch.save({'model_state_dict': self.net.online.state_dict(), 'exploration_rate': self.exploration_rate, 'optimizer_state_dict': self.optimizer.state_dict(), 'step': step}, save_path)
        print(f"Model saved at {save_path}")

    def load(self, load_path):
        checkpoint = torch.load(load_path)
        self.net.online.load_state_dict(checkpoint['model_state_dict'])
        self.net.target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['step']
        print(f"Model loaded from {load_path}")

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}{'TimeDelta':>15}{'Time':>20}\n")
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        self.init_episode()
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        if loss:
            self.curr_ep_loss.append(loss)
        if q:
            self.curr_ep_q.append(q)
        self.curr_ep_length += 1

    def log_episode(self):
        self.moving_avg_ep_rewards.append(self.curr_ep_reward)
        self.moving_avg_ep_lengths.append(self.curr_ep_length)
        self.moving_avg_ep_avg_losses.append(np.mean(self.curr_ep_loss) if self.curr_ep_loss else 0)
        self.moving_avg_ep_avg_qs.append(np.mean(self.curr_ep_q) if self.curr_ep_q else 0)
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = []
        self.curr_ep_q = []

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.moving_avg_ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.moving_avg_ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.moving_avg_ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.moving_avg_ep_avg_qs[-100:]), 3)
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} - MeanReward {mean_ep_reward} - MeanLength {mean_ep_length} - MeanLoss {mean_ep_loss} - MeanQValue {mean_ep_q} - TimeDelta {time_since_last_record}")
        with open(self.save_log, "a") as f:
            f.write(f"{episode:8}{step:8}{epsilon:10.3f}{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}{time_since_last_record:15.3f}{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")
        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env

def load_json_files_in_dir(directory):
    json_files = Path(directory).glob("*.json")
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as file:
            data.append(json.load(file))
    return data

def preprocess(data, transform_size=(84, 84)):
    processed_data = []
    transform = T.Compose([T.ToTensor(), T.Grayscale(), T.Resize(transform_size)])
    for session in data:
        episode = []
        for frame in session:
            frame = np.array(frame)
            frame = transform(frame)
            episode.append(frame)
        processed_data.append(torch.stack(episode))
    return processed_data

def train(mario, logger, max_episodes, max_steps, save_interval, gameplay_sessions):
    for episode in range(max_episodes):
        for session in gameplay_sessions:
            state = session[0]
            for step in range(max_steps):
                action = mario.act(state)
                if step < len(session) - 1:
                    next_state = session[step + 1]
                    reward = 1
                    done = step == len(session) - 2
                else:
                    next_state = state
                    reward = 0
                    done = True
                mario.cache(state, next_state, action, reward, done)
                q, loss = mario.learn()
                logger.log_step(reward, loss, q)
                state = next_state
                if done:
                    break
        logger.log_episode()
        if episode % save_interval == 0:
            mario.save(episode)
        logger.record(episode, mario.exploration_rate, mario.curr_step)

if __name__ == "__main__":
    env = make_env()
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    save_dir = Path("EMOcheckpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    mario = Mario(state_dim, action_dim, save_dir)
    logger = MetricLogger(save_dir)
    max_episodes = 50000
    max_steps = 10000
    save_interval = 500
    gameplay_sessions_dir = "gameplay_sessions"
    gameplay_data = load_json_files_in_dir(gameplay_sessions_dir)
    gameplay_sessions = preprocess(gameplay_data)
    train(mario, logger, max_episodes, max_steps, save_interval, gameplay_sessions)
