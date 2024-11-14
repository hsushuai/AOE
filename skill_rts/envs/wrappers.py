from skill_rts.envs.vec_env import MicroRTSGridModeVecEnv, MicroRTSBotVecEnv
from skill_rts.game.player import Player
from skill_rts.game.game_state import GameState
from skill_rts.game.metric import Metric
from skill_rts.game.trajectory import Trajectory
from skill_rts.envs.record_video import RecordVideo
from skill_rts import logger
import gym
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class MicroRTSLLMEnv(gym.Env):
    """Wrapper Environment for MicroRTS."""
    def __init__(
        self, 
        agents: list,
        max_steps: int=2000,
        map_path: str="maps/8x8/basesWorkers8x8.xml",
        interval: int=100,
        record_video: bool=False,
        run_dir: str="runs",
        display: bool=False,
        payoff_weights: list=[0.1, 0.1],  # alpha, beta
        theme: str="white"
    ):
        """
        Initializes a new instance of the MicroRTS environment with specified settings.

        Args:
            agents (list): A list of agents participating in the environment, length must be 2.
            max_steps (int, optional): Maximum number of steps per episode. Default is 2000.
            map_path (str, optional): Paths to map files for setting up different scenarios.
            interval (int, optional): Update task plan interval for LLM agents.
            record_video (bool, optional): Flag indicating whether to record gameplay video. Default is False.
            run_dir (str, optional): Directory path for saving run log and video recordings. Default is "runs".
            display (bool, optional): Flag indicating whether to display video of the gameplay. Default is False.
            payoff_weights (list, optional): List of weights for calculating the payoff. Default is [0.1, 0.1].
            theme (str, optional): Theme of the game interface; possible values include "white" and "black". Default is "white".
        """
        self.llm_agents = []
        self.bot_agents = []

        self.num_players = 0  # no. of llm agent
        self.max_steps = max_steps
        self.map_path = map_path
        self.interval = interval
        self.record_video = record_video
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.display = display
        self.payoff_weights = payoff_weights
        self.theme = theme

        self.set_agent(agents)
        self._init_env()

        self.time = 0
        self.game_over = True
    
    def _init_env(self):
        reward_weight = np.array([1, 0, 0, 0, 0, 0])
        if self.num_players == 0 and len(self.bot_agents) == 2:
            self.env = MicroRTSBotVecEnv(
                ai1s=[self.bot_agents[0]],
                ai2s=[self.bot_agents[1]],
                max_steps=self.max_steps,
                map_paths=[self.map_path],
                reward_weight=reward_weight,
                autobuild=False
            )
        elif self.num_players == 1 and len(self.bot_agents) == 1:
            self.env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=0,
                num_bot_envs=1,
                max_steps=self.max_steps,
                ai2s=self.bot_agents,
                map_paths=[self.map_path],
                reward_weight=reward_weight,
                autobuild=False
            )
        elif self.num_players == 2 and len(self.bot_agents) == 0:
            self.env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=2,
                num_bot_envs=0,
                max_steps=self.max_steps,
                map_paths=[self.map_path],
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
    
    def reset(self) -> tuple[np.ndarray, list[dict]]:
        """
        Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
            info (optional list): list of a dictionary containing extra information, this is only returned if return_info is set to true.
        """
        return self.env.reset()
    
    def prepare_run(self) -> None:
        self.log_file = open(os.path.join(self.run_dir, "run.log"), "w")
        logger.set_level(logger.INFO)
        logger.set_stream(self.log_file)
        
        raw_obs, raw_info = self.reset()
        self.game_over = False

        self.gs = GameState(raw_info[0]["game_state"])
        self.metric = Metric(self.gs)

        # initialize players
        self.players = []
        for player_id in range(self.num_players):
            self.players.append(Player(player_id, GameState(raw_info[player_id]["player_obs"])))
    
    def step_run(self):
        actions = []
        logger.info((f"{'-'*20} step-{self.time} {'-'*20}"))
        
        for player in self.players:
            if self.time % self.interval == 0:
                tasks = self.llm_agents[player.id].step(player.obs.to_string())
                player.set_tasks(tasks)
            ac = player.step()
            actions.append(ac)
        
        raw_obs, self._raw_payoffs, done, raw_info = self.step(np.array(actions))
        
        self.metric.update(GameState(raw_info[0]["game_state"]))
        for player, info in zip(self.players, raw_info):
            self.gs = GameState(info["game_state"])
            player.update_obs(self.gs)
        
        self.game_over = done[0]
        self.time += 1
    
    def step(self, actions: np.ndarray) -> tuple[list, Trajectory]:
        """Step the environment.

        Accepts an action and returns a tuple (observation, reward, done, info).
        
        Args:
            actions (np.ndarray): actions for each player
        
        Returns:
            obs (list): list of observations for each player
            payoffs (list): list of payoffs for each player
            done (bool): whether the game is over
            info (dict): additional information
        """
        return self.env.step(np.array(actions))
        
    def run(self) -> tuple[list, Trajectory]:
        """Run a complete game.
        
        Returns:
            payoffs (list): list of payoffs for each player
            traj (Trajectory): trajectory of the game
        """
        self.prepare_run()
        while not self.game_over:
            self.step_run()
            print("\r" + " " * 50 + "\r", end="", flush=True)
            print(f"\rRunning step {self.time}", end="", flush=True)
        print("\r" + " " * 50 + "\r", end="", flush=True)
        
        self.end_run()
        self._set_winner()
        self._calculate_payoffs()
        return self.payoffs, self.get_traj()
    
    def _calculate_payoffs(self):
        """Calculate the payoffs for each player."""
        # 10 * win_loss + alpha * damage_dealt + beta * resource_balance
        alpha, beta = self.payoff_weights

        resource_balance = list(map(lambda x, y: x - y, self.metric.resource_spent, self.metric.resource_harvested))
        self.payoffs = list(map(lambda x, y, z: 10 * x + alpha * y + beta * z, self.metric.win_loss, self.metric.damage_dealt, resource_balance))
    
    def _set_winner(self):
        """Set the winner of the game."""
        if self._raw_payoffs[0] > 0:
            self.winner = 0  # player 0 win
        elif self._raw_payoffs[0] < 0:
            self.winner = 1  # player 1 win
        else:
            self.winner = -1  # draw
        self.metric.set_winner(self.winner)
    
    def end_run(self):
        self.env.close()
        self.log_file.close()
    
    def get_traj(self):
        if self.time == 0:
            return None
        else:
            return Trajectory(self.env.get_trajectories())
