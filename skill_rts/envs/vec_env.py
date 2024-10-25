import json
import os
import subprocess
import xml.etree.ElementTree as ET
from itertools import cycle

import gym
import jpype
import jpype.imports
import numpy as np
from jpype.imports import registerDomain
from jpype.types import JArray, JInt


class MicroRTSGridModeVecEnv(gym.Env):
    """VecEnv environment from a microrts environment."""
    def __init__(
        self,
        num_selfplay_envs,
        num_bot_envs,
        partial_obs=False,
        max_steps=2000,
        render_mode="rgb_array",
        frame_skip=0,
        ai2s=[],
        map_paths=[],
        reward_weight=np.array([1, 0, 0, 0, 0, 0]),
        cycle_maps=[],
        autobuild=True,
        jvm_args=[],
    ):
        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        self.num_envs = num_selfplay_envs + num_bot_envs
        assert self.num_bot_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.ai2s = ai2s
        self.map_paths = map_paths
        if len(map_paths) == 1:
            self.map_paths = [map_paths[0] for _ in range(self.num_envs)]
        else:
            assert (len(map_paths) == self.num_envs), "if multiple maps are provided, they should be provided for each environment"
        self.reward_weight = reward_weight

        self.microrts_path = os.path.join(os.getcwd(), "microrts")

        # prepare training maps
        self.cycle_maps = list(map(lambda i: os.path.join(self.microrts_path, i), cycle_maps))
        self.next_map = cycle(self.cycle_maps)

        if autobuild:
            self.build_jar()
        
        # read map
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        self.launch_jvm(jvm_args)
        

        # start microrts client
        from rts.units import UnitTypeTable
        self.real_utt = UnitTypeTable()
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            WinLossRewardFunction,
        )
        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.start_client()

        self.action_space_dims = [6, 4, 4, 4, 4, len(self.utt["unitTypes"]), 7 * 7]
        self.action_space = gym.spaces.MultiDiscrete(np.array([self.action_space_dims] * self.height * self.width).flatten())
        self.action_plane_space = gym.spaces.MultiDiscrete(self.action_space_dims)
        self.source_unit_idxs = np.tile(np.arange(self.height * self.width), (self.num_envs, 1))
        self.source_unit_idxs = self.source_unit_idxs.reshape((self.source_unit_idxs.shape + (1,)))

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
    
    def build_jar(self):
        print(f"removing {self.microrts_path}/microrts.jar...")
        if os.path.exists(f"{self.microrts_path}/microrts.jar"):
            os.remove(f"{self.microrts_path}/microrts.jar")
        print(f"building {self.microrts_path}/microrts.jar...")
        root_dir = os.getcwd()
        print(root_dir)
        subprocess.run(["bash", "build.sh", "&>", "build.log"], cwd=f"{root_dir}")
    
    def launch_jvm(self, jvm_args=[]):
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
            jpype.startJVM(*jvm_args, convertStrings=False)

    def start_client(self):
        from ai.core import AI
        from ts import JNIGridnetVecClient as Client

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = (
            self.vec_client.selfPlayClients[0]
            if len(self.vec_client.selfPlayClients) > 0
            else self.vec_client.clients[0]
        )
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0] * self.num_envs)
        obs = [np.array(ro) for ro in responses.observation]
        return list(zip(obs, np.array(responses.resources)))

    def step_async(self, actions):
        actions = actions.reshape((self.num_envs, self.width * self.height, -1))
        actions = np.concatenate((self.source_unit_idxs, actions), 2)  # specify source unit
        # valid actions
        self.get_action_mask()
        actions = actions[np.where(self.source_unit_mask == 1)]
        action_counts_per_env = self.source_unit_mask.sum(1)
        java_actions = [None] * len(action_counts_per_env)
        action_idx = 0
        for outer_idx, action_count in enumerate(action_counts_per_env):
            java_valid_action = [None] * action_count
            for idx in range(action_count):
                java_valid_action[idx] = JArray(JInt)(actions[action_idx])
                action_idx += 1
            java_actions[outer_idx] = JArray(JArray(JInt))(java_valid_action)
        self.actions = JArray(JArray(JArray(JInt)))(java_actions)

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0] * self.num_envs)
        reward, done = np.array(responses.reward), np.array(responses.done)
        obs = [np.array(ro) for ro in responses.observation]
        infos = [{"raw_rewards": item} for item in reward]
        # check if it is in evaluation, if not, then change maps
        if len(self.cycle_maps) > 0:
            # check if an environment is done, if done, reset the client, and replace the observation
            for done_idx, d in enumerate(done[:, 0]):
                # bot envs settings
                if done_idx < self.num_bot_envs:
                    if d:
                        self.vec_client.clients[done_idx].mapPath = next(self.next_map)
                        response = self.vec_client.clients[done_idx].reset(0)
                        obs[done_idx] = self._encode_obs(np.array(response.observation))
                # selfplay envs settings
                else:
                    if d and done_idx % 2 == 0:
                        done_idx -= self.num_bot_envs  # recalibrate the index
                        self.vec_client.selfPlayClients[done_idx // 2].mapPath = next(self.next_map)
                        self.vec_client.selfPlayClients[done_idx // 2].reset()
                        p0_response = self.vec_client.selfPlayClients[done_idx // 2].getResponse(0)
                        p1_response = self.vec_client.selfPlayClients[done_idx // 2].getResponse(1)
                        obs[done_idx] = self._encode_obs(np.array(p0_response.observation))
                        obs[done_idx + 1] = self._encode_obs(np.array(p1_response.observation))
        return (
            list(zip(obs, np.array(responses.resources))),
            reward @ self.reward_weight,
            done[:, 0],
            infos,
        )

    def step(self, ac):
        self.step_async(ac)
        return self.step_wait()

    def close(self):
        if jpype._jpype.isStarted():
            self.vec_client.close()

    def get_action_mask(self):
        # action_mask shape: [num_envs, map height, map width, 1 + action types + params]
        action_mask = np.array(self.vec_client.getMasks(0))
        # self.source_unit_mask shape: [num_envs, map height * map width * 1]
        self.source_unit_mask = action_mask[:, :, :, 0].reshape(self.num_envs, -1)


class MicroRTSBotVecEnv(MicroRTSGridModeVecEnv):
    def __init__(
        self,
        ai1s=[],
        ai2s=[],
        partial_obs=False,
        max_steps=2000,
        map_paths="maps/10x10/basesTwoWorkers10x10.xml",
        reward_weight=np.array([1, 0, 0, 0, 0, 0]),
        autobuild=True,
        jvm_args=[],
    ):
        self.ai1s = ai1s
        self.ai2s = ai2s
        assert len(ai1s) == len(ai2s), "for each environment, a microrts ai should be provided"
        self.num_envs = len(ai1s)
        self.partial_obs = partial_obs
        self.max_steps = max_steps
        self.map_paths = map_paths
        self.reward_weight = reward_weight

        # read map
        self.microrts_path = os.path.join(os.getcwd(), "microrts")

        if autobuild:
            self.build_jar()

        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

        # launch the JVM
        self.launch_jvm(jvm_args)

        # start microrts client
        from rts.units import UnitTypeTable

        self.real_utt = UnitTypeTable()
        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceCombatUnitRewardFunction,
            ProduceWorkerRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            WinLossRewardFunction,
        )

        self.rfs = JArray(RewardFunctionInterface)(
            [
                WinLossRewardFunction(),
                ResourceGatherRewardFunction(),
                ProduceWorkerRewardFunction(),
                ProduceBuildingRewardFunction(),
                AttackRewardFunction(),
                ProduceCombatUnitRewardFunction(),
                # CloserToEnemyBaseRewardFunction(),
            ]
        )
        self.start_client()

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)

    def start_client(self):
        from ai.core import AI
        from ts import JNIGridnetVecClient as Client

        self.vec_client = Client(
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai1(self.real_utt) for ai1 in self.ai1s]),
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.botClients[0]
        # get the unit type table
        self.utt = json.loads(str(self.render_client.sendUTT()))

    def reset(self):
        responses = self.vec_client.reset([0 for _ in range(self.num_envs)])
        obs = [np.array(ro) for ro in responses.observation]
        return list(zip(obs, np.array(responses.resources)))

    def step_async(self, actions):
        self.actions = JArray(JArray(JArray(JInt)))([JArray(JArray(JInt))([JArray(JInt)([1])])])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_wait(self):
        responses = self.vec_client.gameStep(self.actions, [0 for _ in range(self.num_envs)])
        obs, reward, done = (
            [np.array(ro) for ro in responses.observation],
            np.array(responses.reward),
            np.array(responses.done),
        )
        infos = [{"raw_rewards": item} for item in reward]
        return (
            list(zip(obs, np.array(responses.resources))),
            reward @ self.reward_weight,
            done[:, 0],
            infos,
        )