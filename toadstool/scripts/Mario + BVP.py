import logging
import os
import json
import time
import warnings
import cv2
from matplotlib import pyplot as plt, transforms
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
from nes_py.wrappers import JoypadSpace
from predict_BVP import BVP_CNN
from PIL import Image

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
        transforms = T.Compose([
            T.ToPILImage(),                      # Convert tensor to PIL Image
            T.Resize(self.shape, interpolation=3) # Resize with Image.LANCZOS interpolation
        ])
        observation = transforms(observation)
        observation = T.functional.to_tensor(observation) * 255  # Convert back to tensor and scale to [0, 255]
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

        # Initialize BVP model
        self.bvp_model = BVP_CNN
        self.bvp_model_path = None

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
            #print(state.shape)
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).squeeze(1).unsqueeze(0)
            #print({f'After: {state.shape}'})
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doneT = torch.tensor([done]).to(self.device)
        emotional_rewardT = torch.tensor([emotional_reward]).to(self.device)

        self.memory.add(TensorDict({
            "state": stateT,
            "next_state": next_stateT,
            "action": actionT,
            "reward": rewardT,
            "done": doneT,
            "emotional_reward": emotional_rewardT
        }, batch_size=[]))

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done, emotional_reward = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done", "emotional_reward")
        )
        return state.squeeze(2), next_state.squeeze(2), action.squeeze(), reward.squeeze(), done.squeeze(), emotional_reward.squeeze()
    
    def handle_nan_values(self, tensor):
        # Replace NaN values with 0.0
        tensor[torch.isnan(tensor)] = 0.0
        return tensor
    
    def td_estimate(self, state, action):
        """
        Compute TD estimate (Q-value)
        """
        current_q = self.net(state, model="online")[torch.arange(0, self.batch_size), action]
        return current_q

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

        # if self.curr_step % 500 == 0:
        #     print(f"Step: {self.curr_step}")
        #     print(f"State: {state}")
        #     print(f"Action: {action}")
        #     print(f"Reward: {reward}")     
        new_reward = self.handle_nan_values(reward)   
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(new_reward, emotional_reward, next_state, done)

        # if self.curr_step % 500 == 0:
        #     print(f"TD Estimate: {td_estimate}")
        #     print(f"TD Target: {td_target}")       

        loss = self.update_q_online(td_estimate, td_target)

        return (td_estimate.mean().item(), loss)
    

   
    def load_bvp_model(self, bvp_save_dir):
        checkpoint_path = os.path.join(bvp_save_dir, 'BVP_model_finished.pth')
        
        if os.path.exists(checkpoint_path):
            print(f"Loading model and optimizer state from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        
            model = BVP_CNN().to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            

            
            print(f"Model and optimizer loaded successfully from epoch {checkpoint['epoch']}.")
            model.eval()
            
            self.bvp_model = model
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        

    def save(self, step, bvp_dir):
        """
        Save model parameters and BVP model path.

        Args:
        - step (int): Current step or epoch number for naming the checkpoint file.
        """
        save_path = self.save_dir / f"mario_net_{step}.chkpt"

        # Save MarioNet parameters
        torch.save({
            'model_state_dict': self.net.online.state_dict(),
            'exploration_rate': self.exploration_rate,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'bvp_model_path': bvp_dir
        }, save_path)

        print(f"MarioNet model saved at {save_path}")

    def load(self, load_path):
        """
        Load model parameters and BVP model path.

        Args:
        - load_path (str): Path to the saved checkpoint file.
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.net.online.load_state_dict(checkpoint['model_state_dict'])
        self.net.target.load_state_dict(checkpoint['model_state_dict'])  # Sync target with online
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['step']

        # Load BVP model path if available
        bvp_model_path = checkpoint.get('bvp_model_path', None)
        if bvp_model_path:
            self.bvp_model_path = bvp_model_path
            self.load_bvp_model(bvp_model_path)
            print(f"BVP model path loaded: {self.bvp_model}")
        else:
            print("No BVP model path found in checkpoint.")

        print(f"Model loaded from {load_path}")

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'Finished':>15}{'Score':>15}"
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
        self.high_score = 0
        self.total_game_completions = 0

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

    def log_episode(self, score):
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
        if self.high_score < score:
            self.high_score = score

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
            f"Score {self.high_score} - "
            f"Finished {self.total_game_completions} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{self.total_game_completions:15.3f}"
                f"{self.high_score:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
            plt.clf()
            plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
            plt.legend()
            plt.savefig(getattr(self, f"{metric}_plot"))
    
    def log_game_completion(self, episode):
        self.total_game_completions += 1
        print(f"Episode {episode} - Game Completed - Total Completions: {self.total_game_completions}")

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



def make_env(world, stage, skip=4, shape=(84, 84)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', new_step_api=True)
        else:
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb', apply_api_compatibility=True)
    

    env = JoypadSpace(env, [["right"], ["right", "A"],  ["right", "B"],  ["right", "B","A"]])
    env = SkipFrame(env, skip)
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

env = make_env(1,1)

stage_num = 0
world = 1
score = 0
stage = 1
min_peak_value = 0.1
weighting_factor = 0.5
cumulative_reward = 0.0
bvp_step = 0
score = 0
memory = []
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

# Default values
choice = 'load'
custom_save_dir = "BVP_EMOcheckpoints/2024-07-15T10-33-29/mario_net_8648673.chkpt"
bvp_save_dir = "toadstool/BVPmodels/Finished models/p0"

save_dir = Path("BVP_EMOcheckpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

# Start a new model or load an existing one based on user choice
if choice == 'new':
    mario = start_new_mario(save_dir)
    mario.load_bvp_model(bvp_save_dir)
elif choice == 'load':
    mario = load_existing_mario(custom_save_dir, save_dir)
else:
    raise ValueError("Invalid choice! Please enter 'new' or 'load'.")

logger = MetricLogger(save_dir)
finished = 0
episodes = 20000
for e in range(episodes):
    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        frame = next_state[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure frame is a NumPy array
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        else:
            frame = frame.numpy()

        frame = frame.astype(np.float32)
        # Convert the NumPy array to a PyTorch tensor
        frame_tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)



        # Predict BVP value using the BVP model
        with torch.no_grad():
            bvp_value = mario.bvp_model(frame_tensor).item()

        emotional_reward = calculate_emotional_reward(memory, bvp_value, min_peak_value, weighting_factor)

        # Remember
        mario.cache(state, next_state, action, reward, done, emotional_reward)

        # Learn
        q, loss = mario.learn()

        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            if score < info["score"]:
                score = info["score"]
                  
            if info["flag_get"]:
                stage_num += 1
                if world == 8 and stage == 4:
                    logger.log_game_completion(episode=e)
                world, stage, new_world = make_next_stage(world, stage, stage_num)
                env.close()
                env = make_env(world, stage)
                print(f'{world}-{stage}')
            else:
                env.close()
                env  = make_env(1,1)
                world =1
                stage =1
            break
    logger.log_episode(score)

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step, Finished=finished)

    if (e % 1000 == 0) or (e == episodes - 1):
        mario.save(mario.curr_step, bvp_save_dir)

    env.close()
    env = make_env(1, 1)



