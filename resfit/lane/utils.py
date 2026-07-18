import torch
import numpy as np
import gymnasium as gym
import os
from collections import deque
import random
from torch.utils.data import Dataset
from torch import nn
from data_augs import random_crop


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        batch_size,
        device,
        load_dir="None",
        image_size=84,
        transform=None,
        keep_loaded=False,
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=bool)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self.keep_loaded = keep_loaded
        self.keep_loaded_end = 0

        self.transform_a = None
        self.transform_b = None

        self.demo_starts = None
        self.demo_ends = None

        if load_dir != "None" and load_dir is not None:
            self.load(load_dir)

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        if not self.keep_loaded:
            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        else:
            self.idx += 1
            if self.idx == self.capacity:
                self.idx = self.keep_loaded_end
                self.full = True

    def create_tensors(self, obses, next_obses, actions, rewards, not_dones):
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(actions, device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        not_dones = torch.as_tensor(not_dones, device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_proprio(self):
        idxes = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        return self.create_tensors(
            self.obses[idxes],
            self.next_obses[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.not_dones[idxes],
        )

    def sample_rad(self, aug_funcs, demo_density=None):
        if demo_density is not None:
            assert demo_density <= 1
            assert demo_density >= 0
            demo_batch_size = int(self.batch_size * demo_density)
            exp_batch_size = self.batch_size - demo_batch_size
            demo_idxes = np.random.randint(
                0, self.keep_loaded_end, size=demo_batch_size
            )
            exp_end = self.capacity if self.full else self.idx
            if exp_end * demo_density < self.keep_loaded_end:
                exp_sample_start = 0
            else:
                exp_sample_start = self.keep_loaded_end
            exp_idxes = np.random.randint(
                exp_sample_start, exp_end, size=exp_batch_size
            )
            idxes = np.concatenate([demo_idxes, exp_idxes])
        else:
            idxes = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size
            )

        obses = self.obses[idxes]
        next_obses = self.next_obses[idxes]

        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if "crop" in aug or "cutout" in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)

                if "translate" in aug:
                    obses, tw, th = func(obses)
                    next_obses, _, _ = func(next_obses, tw, th)

        obses, actions, rewards, next_obses, not_dones = self.create_tensors(
            obses,
            next_obses,
            self.actions[idxes],
            self.rewards[idxes],
            self.not_dones[idxes],
        )

        obses = obses / 255.0
        next_obses = next_obses / 255.0

        # augmentations go here
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # skip crop and cutout augs
                if "crop" in aug or "cutout" in aug or "translate" in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_e2c(self):
        idxes = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obs_non_crop = self.obses[idxes]
        next_obs_non_crop = self.next_obses[idxes]

        obses = random_crop(obs_non_crop)
        next_obses = random_crop(next_obs_non_crop)

        obses, actions, rewards, next_obses, not_dones = self.create_tensors(
            obses,
            next_obses,
            self.actions[idxes],
            self.rewards[idxes],
            self.not_dones[idxes],
        )

        obses = obses / 255.0
        next_obses = next_obses / 255.0

        obs_non_crop = torch.as_tensor(obs_non_crop, device=self.device).float() / 255
        next_obs_non_crop = (
            torch.as_tensor(next_obs_non_crop, device=self.device).float() / 255
        )

        return obses, actions, next_obses, obs_non_crop, next_obs_non_crop

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chunks = [c for c in chunks if c[-3:] == ".pt"]
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path, weights_only=False)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end
            self.keep_loaded_end = end
        self.demo_starts = np.load(os.path.join(save_dir, "demo_starts.npy"))
        self.demo_ends = np.load(os.path.join(save_dir, "demo_ends.npy"))

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.capacity if self.full else self.idx, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps
        self.special_reset_save = None

    def reset(self, save_special_steps=False):
        if save_special_steps:
            obs = self.env.reset(save_special_steps=save_special_steps)
        else:
            obs = self.env.reset()
        if save_special_steps:
            self.unpack_special_steps()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def unpack_special_steps(self):
        special_steps_dict = self.env.special_reset_save
        obs_list = special_steps_dict["obs"]
        stacked_obs = []
        for _ in range(self._k):
            self._frames.append(obs_list[0])
        for o in obs_list:
            self._frames.append(o)
            stacked_obs.append(self._get_obs())
        self.special_reset_save = {
            "obs": stacked_obs,
            "act": special_steps_dict["act"],
            "reward": special_steps_dict["reward"],
        }

    def _get_obs(self):
        assert len(self._frames) == self._k
        frames = np.concatenate(list(self._frames), axis=0)
        return frames

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


def create_mlp(in_features, out_features, n_hidden_layers=3, hidden_size=512):
    assert n_hidden_layers > 0
    ff_layers = [
        nn.Linear(in_features=in_features, out_features=hidden_size),
        # nn.ReLU(),
        # nn.SiLU(),
        nn.GELU(),
    ]
    for i in range(1, n_hidden_layers):
        ff_layers += [
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.ReLU(),
            # nn.SiLU(),
            nn.GELU(),
        ]
    ff_layers.append(nn.Linear(in_features=hidden_size, out_features=out_features))
    return nn.Sequential(*ff_layers)
