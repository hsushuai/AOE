from skill_rts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSBotVecEnv
from skill_rts.game.player import Player
from skill_rts.game.game_state import GameState
from skill_rts.game.trajectory import Trajectory
from skill_rts.envs.record_video import RecordVideo
from skill_rts import logger

import gym
import os
import numpy as np

import jpype
import jpype.imports
import numpy as np
from jpype.imports import registerDomain
from jpype.types import JArray, JInt
import xml.etree.ElementTree as ET


class MicroRTSLLMEnv:
    """Wrapper Environment for MicroRTS."""
    def __init__(
        self, 
        agents: list,
        max_steps: int=2000,
        map_paths: list=["maps/8x8/basesWorkers8x8.xml"],
        interval: int=100,
        record_video: bool=False,
        run_dir: str="runs",
        display: bool=False,
        theme: str="white"
    ):
        """
        Initializes a new instance of the MicroRTS environment with specified settings.

        Args:
            agents (list): A list of agents participating in the environment, length must be 2.
            max_steps (int, optional): Maximum number of steps per episode. Default is 2000.
            map_paths (list, optional): List of paths to map files for setting up different scenarios.
            interval (int, optional): Update task plan interval for LLM agents.
            record_video (bool, optional): Flag indicating whether to record gameplay video. Default is False.
            run_dir (str, optional): Directory path for saving run log and video recordings. Default is "runs".
            display (bool, optional): Flag indicating whether to display video of the gameplay. Default is False.
            theme (str, optional): Theme of the game interface; possible values include "white" and "black". Default is "white".
        """
        self.llm_agents = []
        self.bot_agents = []

        self.num_players = 0  # no. of llm agent
        self.max_steps = max_steps
        self.map_paths = map_paths
        self.interval = interval
        self.record_video = record_video
        self.run_dir = run_dir
        if os.path.isdir(run_dir):
            logger.warn(f"Overwriting existing log at '{run_dir}' folder (try specifying a different `run_dir`)")
        os.makedirs(self.run_dir, exist_ok=True)
        self.display = display
        self.theme = theme

        self.set_agent(agents)
        self.init_env()

        self.time = 0
        self.game_over = True
    
    def init_env(self):
        reward_weight = np.array([1, 0, 0, 0, 0, 0])
        if self.num_players == 0 and len(self.bot_agents) == 2:
            self.env = MicroRTSBotVecEnv(
                ai1s=[self.bot_agents[0]],
                ai2s=[self.bot_agents[1]],
                max_steps=self.max_steps,
                map_paths=self.map_paths,
                reward_weight=reward_weight,
                autobuild=False
            )
        elif self.num_players == 1 and len(self.bot_agents) == 1:
            self.env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=0,
                num_bot_envs=1,
                max_steps=self.max_steps,
                ai2s=self.bot_agents,
                map_paths=self.map_paths,
                reward_weight=reward_weight,
                autobuild=False
            )
        elif self.num_players == 2 and len(self.bot_agents) == 0:
            self.env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=2,
                num_bot_envs=0,
                max_steps=self.max_steps,
                map_paths=self.map_paths,
                reward_weight=reward_weight,
                autobuild=False
            )
        else:
            raise ValueError("Couldn't initialize environment base on the given `agents`.")
        
        if self.record_video:
            self.env = RecordVideo(self.env, self.run_dir, display=self.display, theme=self.theme)
    
    def set_agent(self, agents: list):
        assert len(agents) == 2, f"Length of `agents` must be 2, but got a length of {len(agents)}."
        
        self.agents = agents
        for agent in agents:
            if agent.__module__ not in ["skill_rts.agents.bot_agent"]:
                self.num_players += 1  # llm agent
                self.llm_agents.append(agent)
            else:  # java bot
                self.bot_agents.append(agent)
    
    def run(self) -> tuple[list, Trajectory]:
        """Run a complete game."""
        
        log_file = open(os.path.join(self.run_dir, "run.log"), "w")
        logger.set_level(logger.DEBUG)
        logger.set_stream(log_file)
        
        raw_obs, raw_info = self.env.reset()
        self.game_over = False

        players = []
        for player_id in range(self.num_players):
            players.append(Player(player_id, GameState(raw_info[player_id]["player_obs"])))

        while not self.game_over:
            actions = []
            logger.info((f"{'-'*20} step-{self.time} {'-'*20}"))
            for player in players:
                if self.time % self.interval == 0:
                    tasks = self.llm_agents[player.id].step()
                    player.set_tasks(tasks)
                ac = player.step()
                actions.append(ac)
            self.time += 1
            raw_obs, payoffs, done, raw_info = self.env.step(np.array(actions))
            self.game_over = done[0]
        
        self.payoffs = payoffs

        self.env.close()
        log_file.close()
        
        return payoffs, self.get_traj()
    
    def get_traj(self):
        if self.time == 0:
            return None
        else:
            return Trajectory(self.env.get_trajectories()[0])
