import logging
import os
import json
import time
import warnings
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from pathlib import Path
from PIL import Image
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
        transforms = T.Compose([
            T.ToPILImage(),                      # Convert tensor to PIL Image
            T.Resize(self.shape, interpolation=3) # Resize with Image.LANCZOS interpolation
        ])
        observation = transforms(observation)
        observation = T.functional.to_tensor(observation) * 255  # Convert back to tensor and scale to [0, 255]
        return observation.to(torch.uint8)

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


def replay_game_save_frames(env, session_path, output_directory, render_screen=False):
    with open(session_path) as json_file:
        data = json.load(json_file)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    next_state = env.reset()
    steps = 0
    stage_num = 0
    world = 1
    stage = 1
    finish = False

    for i, action in enumerate(data["obs"]):
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



                # Convert PyTorch tensor to numpy array
        next_state_np = next_state.squeeze(0).cpu().numpy().astype(np.uint8)


        # Save frame as image with world and stage-specific name
        frame_image = Image.fromarray(next_state_np)
        frame_name = f"frame_{i}.png"
        frame_path = os.path.join(output_directory, frame_name)
        frame_image.save(frame_path)

        if info.get("flag_get"):
            finish = True

        steps+=1

        if done:

            if finish or steps >= 16000:
                stage_num += 1
                world, stage, new_world = make_next_stage(world, stage, stage_num)
                env.close()
                torch.cuda.empty_cache()
                env = make_env(world, stage)
                steps = 0
                finish = False
            
            next_state = env.reset()

            state = next_state




def make_env(world, stage, skip=4, shape=(84, 84)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if gym.__version__ < '0.26':
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', new_step_api=True)
        else:
            env = gym_super_mario_bros.make(f'SuperMarioBros-{world}-{stage}-v0', render_mode='rgb_array', apply_api_compatibility=True)
            print(f'{world}-{stage}')
   # env = SkipFrame(env, skip)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    return env




# Main code to use data loader and DDQN

data_dir = "toadstool/participants"
toadstool_data = toadstool_data_loader.load_participant_data(data_dir)
single_participant = toadstool_data_loader.load_single_participant(data_dir, 6)

# Initialize DDQN components
save_dir = Path("EMOcheckpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
env = make_env(1,1)

choice = 'new'
action = 'learn'
custom_save_dir = "EMOcheckpoints/2024-07-07T10-33-01/mario_net_4605020.chkpt"


session_path = os.path.join(data_dir, single_participant["participant_id"], f"{single_participant['participant_id']}_session.json")
output_directory = "toadstool/images/par_6"
replay_game_save_frames(env, session_path, output_directory, render_screen=False)






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
