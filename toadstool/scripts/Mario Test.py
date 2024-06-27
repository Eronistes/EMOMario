import os
import json
import time
import warnings
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from pathlib import Path

from tensordict import TensorDict
import toadstool_data_loader
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from torchvision import transforms as T
import gym_super_mario_bros
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import replay_game_session
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
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
        # permute [H, W, C] array to [C, H, W] tensor
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
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation



# Define MarioNet class with build_cnn method
class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        # Build CNN architecture for both online and target networks
        self.online = self.build_cnn(c, h, w, output_dim)
        self.target = self.build_cnn(c, h, w, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Freeze target network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def build_cnn(self, c, h, w, output_dim):
        """
        Build the CNN architecture
        """
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


# Define Mario class incorporating DDQN algorithm components
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.save_dir = Path(save_dir)
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): print(f'Using:  {self.device}')

        # Initialize MarioNet
        self.net = MarioNet(state_dim, action_dim)
        self.net = self.net.to(device=self.device)

        # Initialize exploration parameters
        self.exploration_rate = 0.1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Initialize replay buffer
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        print(f'memory: {self.device}')
        self.batch_size = 32

        # Initialize DDQN parameters
        self.gamma = 0.9
        self.lr = 0.00025
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Initialize synchronization parameters
        self.sync_every = 10000  # Synchronize target network every 10000 steps
        self.burnin = 10000  # Start learning after 10000 steps
        self.learn_every = 4  # Update the online network every 4 steps

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])


        self.memory.add(TensorDict({
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done
        }, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        Compute TD estimate (Q-value)
        """
        current_q = self.net(state, model="online")[torch.arange(0, self.batch_size), action]
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_q_online(self, td_estimate, td_target):
        """
        Update online Q-network
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_q_target(self):
        """
        Synchronize target Q-network with online Q-network
        """
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """
        Update Q-network parameters using DDQN algorithm
        """
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
        """
        Save model parameters
        """
        save_path = self.save_dir / f"mario_net_{step}.chkpt"
        torch.save({
            'model_state_dict': self.net.online.state_dict(),
            'exploration_rate': self.exploration_rate,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step
        }, save_path)
        print(f"Model saved at {save_path}")

    def load(self, load_path):
        """
        Load model parameters
        """
        checkpoint = torch.load(load_path)
        self.net.online.load_state_dict(checkpoint['model_state_dict'])
        self.net.target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['step']
        print(f"Model loaded from {load_path}")

    def observe(self, states, actions, rewards, dones, next_states):
        """
        Store a batch of observed experiences to the replay buffer

        Inputs:
        states (list of ``LazyFrame``),
        actions (list of ``int``),
        rewards (list of ``float``),
        dones (list of ``bool``),
        next_states (list of ``LazyFrame``)
        """
        for state, action, reward, done, next_state in zip(states, actions, rewards, dones, next_states):
            self.cache(state, next_state, action, reward, done)

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'Finished':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []
        self.times_finished = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1
        time = 0
        time+1
        if time%20000 == 0:
            print(f'Reward: {self.curr_ep_reward}')

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        print("episode log ")
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step, Finished):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Finished {Finished} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{Finished:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))

_STAGE_ORDER = [
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (1, 4),
    (3, 1),
    (4, 1),
    (2, 1),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2)
]

def make_next_stage(world, stage, num):
    if num < len(_STAGE_ORDER):
        world = _STAGE_ORDER[num][0]
        stage = _STAGE_ORDER[num][1]
    else:
        if stage >= 4:
            stage = 1
            if world >= 8:
                world = 1
            else:
                world += 1
        else:
            stage += 1

    return world, stage, f"SuperMarioBros-{world}-{stage}-v0"





def replay_game_from_actions(env, mario, session_path, logger, render_screen=False):
    with open(session_path) as json_file:
        data = json.load(json_file)

    next_state = env.reset()
    steps = 0
    stage_num = 0
    world = 1
    stage = 1
    finish = False

    # Initialize variables to store skipped frames for learning
    skip = 4
    stacked_frames = deque(maxlen=skip)
    state = next_state

    for action in data["obs"]:
        if render_screen:
            env.render()

        result = env.step(action)

        # Check the length of the result to unpack correctly
        if len(result) == 4:
            next_state, reward, done, info = result  # Old API format
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result  # New API format
            done = terminated or truncated
        else:
            raise ValueError("Unexpected result format from env.step(action)")

        done = done or info.get('flag_get', False)

        steps += 1

        # Cache the current experience
        mario.cache(state, next_state, action, reward, done)
        mario.act(state)
        state = next_state

        # Perform learning step
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)

        if info.get("flag_get"):
            finish = True

        if done:
            if finish or steps >= 16000:
                stage_num += 1
                world, stage, new_world = make_next_stage(world, stage, stage_num)
                print( f"Step {mario.curr_step} - ")
                env.close()
                torch.cuda.empty_cache()
                env = make_env(world, stage)
                finish = False
                steps = 0
            
            next_state = env.reset()
            state = next_state

    env.close()
    torch.cuda.empty_cache()


def make_env(world, stage, skip=4, shape=(84, 84)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', new_step_api=True)
        else:
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='human', apply_api_compatibility=True)
            print(f'{world}-{stage}')
   # env = SkipFrame(env, skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    env = gym.wrappers.FrameStack(env, skip)
    return env

# Function to start a new Mario instance
def start_new_mario(save_dir):
    return Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

# Function to load an existing Mario instance
def load_existing_mario(load_dir, save_dir):
    mario_instance = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    mario_instance.load(load_dir)
    return mario_instance


# Main code to use data loader and DDQN

data_dir = "toadstool/participants"
toadstool_data = toadstool_data_loader.load_participant_data(data_dir)
single_participant = toadstool_data_loader.load_single_participant(data_dir, 0)

# Initialize DDQN components
save_dir = Path("EMOcheckpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
env = make_env(1,1)

choice = 'load'
action = 'ai'
custom_save_dir = "EMOcheckpoints/2024-06-25T11-12-28/mario_net_5725160.chkpt"

# Start a new model or load an existing one based on user choice
if choice == 'new':
    mario = start_new_mario(save_dir)
elif choice == 'load':
    mario = load_existing_mario(custom_save_dir, save_dir)
else:
    raise ValueError("Invalid choice! Please enter 'new' or 'load'.")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if action == 'learn':
    logger = MetricLogger(save_dir)

    # Training loop
    finished = 0
    episodes = 12000
    session_path = os.path.join(data_dir, single_participant["participant_id"], f"{single_participant['participant_id']}_session.json")
    for e in range(episodes):
        print(f"Replaying session for participant: {single_participant['participant_id']}, Episode: {e+1}/{episodes}")
        replay_game_from_actions(env, mario, session_path, logger, render_screen=False)
        env = make_env(1,1)
        logger.log_episode()
        mario.save(mario.curr_step)
        print(
        f"Episode {e} - "
        f"Step {mario.curr_step} - "
        f"Epsilon {mario.exploration_rate} - ")
        if e % 5 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step, Finished=finished)

    print("Training completed.")

elif action == 'ai':
    state = env.reset()
    done = False
    cumulative_reward = 0.0
    episodes = 100

    for e in range(episodes):


        while not done:
            action = mario.act(state)  # Choose action based on the trained policy
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result  # Old API format
            elif len(result) == 5:
                next_state, reward, terminated, truncated, info = result  # New API format
                done = terminated or truncated
            else:
                raise ValueError("Unexpected result format from env.step(action)")
            cumulative_reward += reward
            state = next_state  # Update current state to the next state

            # Optionally render the environment to watch the agent's behavior
            #env.render()

        print(f"Total Cumulative Reward: {cumulative_reward}")
        env.close()
        env = make_env(1,1)



"""# Training loop all participants
finished = 0
episodes = 12000
for e in range(episodes):
    for participant_data in toadstool_data:
        session_path = os.path.join(data_dir, participant_data["participant_id"], f"{participant_data['participant_id']}_session.json")


        print(f"Replaying session for participant: {participant_data['participant_id']}, Episode: {e+1}/{episodes}")
        replay_game_from_actions(env, mario, session_path, logger, render_screen=False)
        env = make_env(1,1)
    logger.log_episode()
    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step, Finished=finished)

print("Training completed.")"""
