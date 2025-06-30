#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from robosuite.wrappers import Wrapper
from torch import Tensor

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.utils.utils import get_channel_first_image_shape
from lerobot.configs.types import FeatureType, PolicyFeature


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations["observation.environment_state"] = env_state

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = torch.from_numpy(observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)
    return_observations["observation.state"] = agent_pos

    return return_observations


def env_to_policy_features(env_cfg: EnvConfig) -> dict[str, PolicyFeature]:
    # TODO(aliberts, rcadene): remove this hardcoding of keys and just use the nested keys as is
    # (need to also refactor preprocess_observation and externalize normalization from policies)
    policy_features = {}
    for key, ft in env_cfg.features.items():
        if ft.type is FeatureType.VISUAL:
            if len(ft.shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={ft.shape})")

            shape = get_channel_first_image_shape(ft.shape)
            feature = PolicyFeature(type=ft.type, shape=shape)
        else:
            feature = ft

        policy_key = env_cfg.features_map[key]
        policy_features[policy_key] = feature

    return policy_features


def are_all_envs_same_type(env: gym.vector.VectorEnv) -> bool:
    first_type = type(env.envs[0])  # Get type of first env
    return all(type(e) is first_type for e in env.envs)  # Fast type check


def check_env_attributes_and_types(env: gym.vector.VectorEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("once", UserWarning)  # Apply filter only in this function

        if not (hasattr(env.envs[0], "task_description") and hasattr(env.envs[0], "task")):
            warnings.warn(
                "The environment does not have 'task_description' and 'task'. Some policies require these features.",
                UserWarning,
                stacklevel=2,
            )
        if not are_all_envs_same_type(env):
            warnings.warn(
                "The environments have different types. Make sure you infer the right task from each environment. Empty task will be passed instead.",
                UserWarning,
                stacklevel=2,
            )


def add_envs_task(env: gym.vector.VectorEnv, observation: dict[str, Any]) -> dict[str, Any]:
    """Adds task feature to the observation dict with respect to the first environment attribute."""
    if hasattr(env.envs[0], "task_description"):
        observation["task"] = env.call("task_description")
    elif hasattr(env.envs[0], "task"):
        observation["task"] = env.call("task")
    else:  #  For envs without language instructions, e.g. aloha transfer cube and etc.
        num_envs = observation[list(observation.keys())[0]].shape[0]
        observation["task"] = ["" for _ in range(num_envs)]
    return observation


class GymWrapper(Wrapper, gym.Env):
    """
    This file implements a wrapper for facilitating compatibility with OpenAI gym.
    This is useful when using these environments with code that assumes a gym-like
    interface.
    """

    metadata = {"render_modes": [], "render_fps": 20}
    render_mode = None
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module
    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.
    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()

        # Whether to flatten the observation space
        self.flatten_obs: bool = flatten_obs

        # TODO(branyang02): remove flatten_obs option in the future
        if self.flatten_obs:
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:

            def get_box_space(sample):
                """Util fn to obtain the space of a single numpy sample data"""
                if np.issubdtype(sample.dtype, np.integer):
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    low = float("-inf")
                    high = float("inf")
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            # TODO(branyang02): create better way to handle adjusted observation space
            # self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "agentview_image": get_box_space(obs["agentview_image"]),
                            "robot0_eye_in_hand_image": get_box_space(obs["robot0_eye_in_hand_image"]),
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            obs["robot0_eef_pos"].size
                            + obs["robot0_eef_quat"].size
                            + obs["robot0_gripper_qpos"].size,
                        ),
                        dtype=np.float64,
                    ),
                }
            )

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def _filter_obs(self, obs_dict) -> dict:
        """
        Filters keys of interest out of the observation dictionary, returning a filterd dictionary.
        """

        # TODO(branyang02): remove hard-coded keys
        # return {key: obs_dict[key] for key in self.keys if key in obs_dict}
        obs = {}
        pixels = {}
        agent_pos = {}
        for key in self.keys:
            if key in obs_dict:
                if "image" in key:
                    pixels[key] = obs_dict[key]
                elif "eef_pos" in key or "eef_quat" in key or "gripper_qpos" in key:
                    agent_pos[key] = obs_dict[key]

        obs["pixels"] = pixels
        obs["agent_pos"] = np.concatenate(list(agent_pos.values()), axis=-1)

        return obs

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation instead of normal OrderedDict and optionally resets seed
        Returns:
            2-tuple:
                - (np.array) observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        self._last_obs = ob_dict
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation instead of normal OrderedDict.
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        self._last_obs = ob_dict
        obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward
        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]
        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        wrapper for calling underlying env close function
        """
        self.env.close()

    def __getattr__(self, attr):
        if attr == "_max_episode_steps":
            return self.env.horizon
        super().__getattr__(attr)

    def render(self):
        return self._last_obs["agentview_image"]
