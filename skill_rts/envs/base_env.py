import gym
import json
import os
import numpy as np
from abc import ABC, abstractmethod
import jpype
import jpype.imports
from jpype.imports import registerDomain
from jpype.types import JArray, JInt
import xml.etree.ElementTree as ET
from PIL import Image


class VecEnv(ABC, gym.Env):
    def __init__(
        self,
        map_path: str,
        max_steps: int,
        reward_weight: np.ndarray,
        partial_obs: bool
    ):
        self.map_path = map_path
        self.max_steps = max_steps
        self.reward_weight = reward_weight
        self.partial_obs = partial_obs

        self.microrts_path = os.path.join(os.getcwd(), "microrts")
        self.launch_jvm()

        # read map
        root = ET.parse(os.path.join(self.map_path)).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # start microrts client
        self._start_client()

        # action_space_dims = [6, 4, 4, 4, 4, 7, 7 * 7]
        # self.action_space = gym.spaces.MultiDiscrete(np.array([action_space_dims] * self.height * self.width).flatten())
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
    
    def launch_jvm(self):
        if not jpype._jpype.isStarted():
            registerDomain("ts", alias="tests")
            registerDomain("ai")
            jars = [
                "microrts.jar",
                "lib/bots/Coac.jar",
                "lib/bots/Droplet.jar",
                "lib/bots/GRojoA3N.jar",
                "lib/bots/Izanagi.jar",
                "lib/bots/MixedBot.jar",
                "lib/bots/TiamatBot.jar",
                "lib/bots/UMSBot.jar",
                "lib/bots/mayariBot.jar",  # "MindSeal.jar"
            ]
            
            for jar in jars:
                jpype.addClassPath(os.path.join(self.microrts_path, jar))
            jpype.startJVM(convertStrings=False)
    
    @abstractmethod
    def start_client(self):
        """Initialize and start the client.

        This method should be implemented by subclasses to establish a connection
        to the necessary services or resources. After calling this method, the
        `self.client` attribute should be set up and ready for use.
        """
        pass

    def _start_client(self):
        self.start_client()
        if not self.client:
            raise RuntimeError("Failed to initialize client: 'self.client' is None.")

    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
            info (dict list): list of a dictionary containing extra information, this is only returned if return_info is set to true
        """
        response = self.client.reset(0)
        return self._parse_responses(response, True)

    def step(self, actions) -> tuple[np.ndarray, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (observations, rewards, dones, infos) for each player.

        Args:
            actions (np.ndarray): actions provided by the agents

        Returns:
            observations (np.ndarray): list of observations from the environment
            rewards (float): list of rewards received after taking the action
            done (bool): list of booleans indicating if each episode has ended
            info (dict): list of dictionaries information containing raw game state and player raw observation
        """
        response = self.client.gameStep(self.actions, 0)
        observation, reward, done, info = self._parse_responses(response)
        return (
            observation,
            reward @ self.reward_weight,
            done[0],
            info,
        )
    
    def _encode_obs(self, obs):
        obs = obs.reshape(len(obs), -1).clip(0, np.array([self.num_planes]).T - 1)
        obs_planes = np.zeros((self.height * self.width, self.num_planes_prefix_sum[-1]), dtype=np.int32)
        obs_planes_idx = np.arange(len(obs_planes))
        obs_planes[obs_planes_idx, obs[0]] = 1

        for i in range(1, self.num_planes_len):
            obs_planes[obs_planes_idx, obs[i] + self.num_planes_prefix_sum[i]] = 1
        return obs_planes.reshape(self.height, self.width, -1)
    
    def _parse_responses(self, response, is_reset=False):
        observation = self._encode_obs(np.array(response.observation))
        raw_info = response.info
        info = {
            "game_state": json.loads(str(raw_info[0])) if raw_info[0] is not None else None,
            "player_obs": json.loads(str(raw_info[1])) if raw_info[1] is not None else None
        }
        if is_reset:
            return observation, info
        reward, done = np.array(response.reward), np.array(response.done)
        return observation, reward, done, info

    def close(self):
        if jpype._jpype.isStarted():
            self.client.close()
            jpype.shutdownJVM()
    
    def render(self, mode: str="rgb_array"):
        bytes_array = np.array(self.client.render(self.display, self.theme))
        image = Image.frombytes("RGB", (640, 640), bytes_array)
        return np.array(image)[:, :, ::-1]