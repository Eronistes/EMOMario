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
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        # Initialize replay buffer
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=self.device))
        self.batch_size = 32

        # Initialize DDQN parameters
        self.gamma = 0.9
        self.lr = 0.00025
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Initialize synchronization parameters
        self.sync_every = 10000  # Synchronize target network every 10000 steps
        self.burnin = 1000  # Start learning after 10000 steps
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


    def cache(self, state, next_state, action, reward, done, emotional_reward):
        state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        next_state = next_state[0].__array__() if isinstance(next_state, tuple) else next_state.__array__()

        stateT = torch.tensor(state).to(self.device)
        next_stateT = torch.tensor(next_state).to(self.device)
        actionT = torch.tensor([action]).to(self.device)
        rewardT = torch.tensor([reward]).to(self.device)
        doneT = torch.tensor([done]).to(self.device)
        emotional_rewardT = torch.tensor([emotional_reward]).to(self.device)

        self.memory.add(TensorDict({
            "state": stateT,
            "next_state": next_stateT,
            "action": actionT,
            "reward": rewardT,
            "done": doneT,
            "emotional_reward": emotional_rewardT
        }))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done, emotional_reward = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done", "emotional_reward")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze(), emotional_reward.squeeze()

    @torch.no_grad()
    def td_target(self, reward, emotional_reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[torch.arange(0, self.batch_size), best_action]
        
        # Combine extrinsic and emotional rewards with a weighting factor
        total_reward = reward + emotional_reward
        td_target = total_reward + (1 - done.float()) * self.gamma * next_Q
        return td_target.float()

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
        
        state, next_state, action, reward, done, emotional_reward = self.recall()

        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(reward, emotional_reward, next_state, done)

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


    def update_policy(self, cumulative_reward, logger):
        """
        Update policy based on cumulative reward using Q-learning update.

        Inputs:
        cumulative_reward (float): The cumulative reward accumulated during the replay session.
        logger (MetricLogger): Object for logging metrics like reward and loss.
        """
        # Decay exploration rate (optional, if using epsilon-greedy)
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # Update Q-values based on cumulative reward
        alpha = self.lr  # Learning rate (could be different from self.lr if desired)

        # Sample a batch of experiences from memory
        state, next_state, action, reward, done = self.recall()

        # Compute TD targets and estimates for the entire batch
        next_state_Q = self.net(next_state, model="online")
        best_actions = torch.argmax(next_state_Q, axis=1)
        next_Q_values = self.net(next_state, model="target")[torch.arange(self.batch_size), best_actions]
        td_targets = reward + (1 - done.float()) * self.gamma * next_Q_values

        current_Q_values = self.net(state, model="online")[torch.arange(self.batch_size), action]
        td_estimates = current_Q_values

        # Calculate loss using Smooth L1 Loss (Huber loss)
        loss = self.loss_fn(td_estimates, td_targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # Synchronize target network periodically (optional)
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()

        return loss.item()



class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = Path(save_dir) / "log"
        with open(self.save_log, "w") as f:
            f.write(
                "Episode,Step,AvgReward,MaxReward,MinReward,StdReward,AvgLoss,MaxLoss,MinLoss,StdLoss,AvgQ,MaxQ,MinQ,StdQ\n"
            )

        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)
        self.ep_avg_losses = deque(maxlen=100)
        self.ep_avg_qs = deque(maxlen=100)

        self.episode_data = []
        self.reset_episode_metrics()

        self.record_time = time.time()
        self.plot_dir = Path(save_dir) / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def reset_episode_metrics(self):
        self.metrics = {
            'avg_reward': 0,
            'max_reward': 0,
            'min_reward': 0,
            'std_reward': 0,
            'avg_loss': 0,
            'max_loss': 0,
            'min_loss': 0,
            'std_loss': 0,
            'avg_q': 0,
            'max_q': 0,
            'min_q': 0,
            'std_q': 0,
        }

    def log_step(self, reward, loss, q):
        self.episode_data.append({
            'reward': reward,
            'loss': loss if loss is not None else 0,
            'q': q if q is not None else 0
        })


    def log_episode(self, episode, step):
        if not self.episode_data:
            return self.metrics

        rewards = [data['reward'] for data in self.episode_data]
        losses = [data['loss'] for data in self.episode_data]
        qs = [data['q'] for data in self.episode_data]

        self.metrics['avg_reward'] = np.mean(rewards)
        self.metrics['max_reward'] = np.max(rewards)
        self.metrics['min_reward'] = np.min(rewards)
        self.metrics['std_reward'] = np.std(rewards)


        self.metrics['avg_loss'] = np.mean(losses)
        self.metrics['max_loss'] = np.max(losses)
        self.metrics['min_loss'] = np.min(losses)
        self.metrics['std_loss'] = np.std(losses)


        self.metrics['avg_q'] = np.mean(qs)
        self.metrics['max_q'] = np.max(qs)
        self.metrics['min_q'] = np.min(qs)
        self.metrics['std_q'] = np.std(qs)

        self.ep_rewards.append(self.metrics['avg_reward'])
        self.ep_lengths.append(len(self.episode_data))
        self.ep_avg_losses.append(self.metrics['avg_loss'])
        self.ep_avg_qs.append(self.metrics['avg_q'])

        self._log_to_file(episode, step)
        self.episode_data = []

        return self.metrics

    def _log_to_file(self, episode, step):
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode},{step},{self.metrics['avg_reward']},{self.metrics['max_reward']},{self.metrics['min_reward']},{self.metrics['std_reward']},"
                f"{self.metrics['avg_loss']},{self.metrics['max_loss']},{self.metrics['min_loss']},{self.metrics['std_loss']},"
                f"{self.metrics['avg_q']},{self.metrics['max_q']},{self.metrics['min_q']},{self.metrics['std_q']}\n"
            )

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.mean(self.ep_rewards)
        mean_ep_length = np.mean(self.ep_lengths)
        mean_ep_loss = np.mean(self.ep_avg_losses)
        mean_ep_q = np.mean(self.ep_avg_qs)

        print(f"Episode {episode} - Step {step} - "
              f"Mean Reward: {mean_ep_reward:.2f} - "
              f"Mean Length: {mean_ep_length:.2f} - "
              f"Mean Loss: {mean_ep_loss:.2f} - "
              f"Mean Q Value: {mean_ep_q:.2f} - "
              f"Epsilon: {epsilon:.2f}")

        # Reset episode metrics after logging
        self.plot_metrics()
        self.reset_episode_metrics()


    def plot_metrics(self):
        self._plot_metric(self.ep_rewards, 'Avg Reward', 'Reward')
        self._plot_metric(self.ep_avg_losses, 'Avg Loss', 'Loss')
        self._plot_metric(self.ep_avg_qs, 'Avg Q Value', 'Q Value')

    def _plot_metric(self, metric, title, ylabel):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(metric)), metric, label=title)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.legend()

        plot_path = self.plot_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"Updated plot for {title} at {plot_path}")

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


def calculate_emotional_reward(memory, bvp_amplitude, min_peak_value, weighting_factor):
    """
    Calculate emotional reward for a given memory in the DQN.

    Parameters:
    - memory (list): A list of dictionaries representing game frames stored in memory.
    - bvp_amplitude (np.ndarray): Normalized BVP amplitude (amp') associated with the memory.
    - min_peak_value (float): Minimum peak value used in BVP amplitude preprocessing.
    - weighting_factor (float): Weighting factor (W) between 0 and 1, to adjust emotional reward emphasis.

    Returns:
    - total_reward (float): Total emotional reward for the memory.
    """
    # Calculate intrinsic emotional reward (Ri)
    amp0 = (bvp_amplitude - min_peak_value) / (1 - min_peak_value)
    Ri = (amp0 - 0.5) * 5

    # Initialize total emotional reward
    total_emotional_reward = 0.0

    # Combine intrinsic emotional reward with extrinsic reward for each frame in memory
    for frame in memory:
        total_emotional_reward += Ri

    # Calculate total combined reward using the weighting factor (W)
    Re = np.sum([frame['reward'] for frame in memory])  # Extrinsic reward sum for all frames
    total_reward = Re * (1 - weighting_factor) + total_emotional_reward * weighting_factor

    return total_reward


def replay_game_from_actions(env, mario = Mario, session_path = str, logger = MetricLogger, bvp_data = np.ndarray, render_screen=False):
    with open(session_path) as json_file:
        data = json.load(json_file)

    next_state = env.reset()
    steps = 0
    stage_num = 0
    world = 1
    stage = 1
    finish = False
    min_peak_value = 0.1
    weighting_factor = 0.5

    # Initialize variables to store skipped frames for learning
    skip = 4
    stacked_frames = deque(maxlen=skip)
    state = next_state
    cumulative_reward = 0.0
    for action in data["obs"]:
        if render_screen:
            env.render()

        result = env.step(action)
        mario.act(state)


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

        cumulative_reward += reward

         # Calculate BVP amplitude (amp') for the current state/frame 
        bvp_amplitude = bvp_data[steps]


        # Calculate emotional reward using the custom function
        emotional_reward = calculate_emotional_reward([{'state': state}], bvp_amplitude, min_peak_value, weighting_factor)


        # Cache the current experience
        mario.cache(state, next_state, action, reward, done, emotional_reward)
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
                mario.update_policy(cumulative_reward, logger)
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
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb', apply_api_compatibility=True)
            print(f'{world}-{stage}')
   # env = SkipFrame(env, skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    #env = gym.wrappers.FrameStack(env, skip)
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
single_participant = toadstool_data_loader.load_single_participant(data_dir, 1)

# Initialize DDQN components
save_dir = Path("EMOcheckpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
env = make_env(1,1)

choice = 'new'
action = 'learn'
custom_save_dir = "EMOcheckpoints/2024-06-30T10-39-29/mario_net_21282660.chkpt"

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
        replay_game_from_actions(env, mario, session_path, logger, single_participant['BVP'],  render_screen=False)
        env = make_env(1,1)
        logger.log_episode(e, mario.curr_step)
        mario.save(mario.curr_step)
        print(
        f"Episode {e} - ")
        if e % 5 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

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
        env.reset()
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
