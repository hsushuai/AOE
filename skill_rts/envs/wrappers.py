from skill_rts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSBotVecEnv
from skill_rts.game.player import Player
from skill_rts.game.observation import Observation
from skill_rts.envs.record_video import RecordVideo
from skill_rts import logger

import os
import numpy as np


class MicroRTSLLMEnv:
    """Wrapper Environment for MicroRTS."""
    def __init__(
        self, 
        agents: list,
        max_steps: int=2000,
        map_paths: list=["maps/8x8/basesWorkers8x8.xml"],
        record_video: bool=False,
        run_dir: str="runs",
        show_video: bool=False,
        theme: str="white"
    ):
        """
        Initializes a new instance of the MicroRTS environment with specified settings.

        Args:
            agents (list): A list of agents participating in the environment, length must be 2.
            max_steps (int, optional): Maximum number of steps per episode. Default is 2000.
            map_paths (list, optional): List of paths to map files for setting up different scenarios.
            record_video (bool, optional): Flag indicating whether to record gameplay video. Default is False.
            run_dir (str, optional): Directory path for saving run log and video recordings. Default is "runs".
            show_video (bool, optional): Flag indicating whether to display video of the gameplay. Default is False.
            theme (str, optional): Theme of the game interface; possible values include "white" and "black". Default is "white".
        """
        self.llm_agents = []
        self.bot_agents = []

        self.num_players = 0  # no. of llm agent
        self.max_steps = max_steps
        self.map_paths = map_paths
        self.record_video = record_video
        self.run_dir = run_dir
        if os.path.isdir(run_dir):
            logger.warn(f"Overwriting existing log at '{run_dir}' folder (try specifying a different `run_dir`)")
        os.makedirs(self.run_dir, exist_ok=True)
        self.show_video = show_video
        self.theme = theme

        self.set_agent(agents)
        self.init_env()

        self.timestep = 0
        self.is_over = True
    
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
            self.env = RecordVideo(self.env, self.run_dir, show=self.show_video, theme=self.theme)
    
    def set_agent(self, agents: list):
        assert len(agents) == 2, f"Length of `agents` must be 2, but got a length of {len(agents)}."
        
        self.agents = agents
        for agent in agents:
            if agent.__module__ not in ["skill_rts.agents.bot_agent"]:
                self.num_players += 1  # llm agent
                self.llm_agents.append(agent)
            else:  # java bot
                self.bot_agents.append(agent)
    
    def run(self):
        """Run a complete game."""
        
        log_file = open(os.path.join(self.run_dir, "run.log"), "w")
        logger.set_level(logger.DEBUG)
        logger.set_stream(log_file)
        
        raw_obs = self.env.reset()
        self.is_over = False

        while not self.is_over:
            actions = []
            logger.info((f"{'-'*20} step-{self.timestep} {'-'*20}"))
            for player_id in range(self.num_players):
                obs = Observation(raw_obs[player_id])
                player = Player(player_id, obs)
                tasks = self.llm_agents[player_id].step()
                ac = player.step(self._parse_task(tasks))
                actions.append(ac)
            self.timestep += 1
            raw_obs, payoffs, done, info = self.env.step(np.array(actions))
            self.is_over = done[0]
        
        self.payoffs = payoffs

        self.env.close()
        log_file.close()
        
        return payoffs
    
    def _parse_task(self, text: str) -> list:
        import ast
        import re

        task_list = []
        params_list = []
        text = text.split("START OF TASK")[1].split("END OF TASK")[0]
        text_list = text.split("\n")
        for task_with_params in text_list:
            task_beg = task_with_params.find("[")
            task_end = task_with_params.find("]")
            param_beg = task_with_params.find("(")
            param_end = task_with_params.rfind(")")
            if task_beg + 1 and task_end + 1:
                task = task_with_params[task_beg : task_end + 1]
            else:
                task = None
            params = re.sub(
                r"(?<!\')(\b[a-zA-Z_]+\b)(?!\')",
                r"'\1'",
                task_with_params[param_beg : param_end + 1],
            )
            params = re.sub(r"'(\d+)'", r"\1", params)
            if param_beg + 1 and param_end + 1:
                params = ast.literal_eval(params)
                if task is not None:
                    task_list.append(task)
                    params_list.append(params)
        logger.info("Parsed Tasks from LLM's Respond:")
        for task, params in zip(task_list, params_list):
            logger.info(f"{task}{params}")
        return list(zip(task_list, params_list))
    