from skill_rts.envs import MicroRTSLLMEnv
from skill_rts.agents import bot_agent
from ace.agent.agent import Planner
from ace.configs.config import cfg
from skill_rts import logger

logger.set_level(logger.DEBUG)


def get_players():
    players_cfg = cfg["players"]
    players = []
    for i, player_cfg in enumerate(players_cfg):
        if player_cfg["model"] in bot_agent.ALL_AIS:
            players.append(bot_agent.get_agent(player_cfg["model"]))
        else:
            players.append(Planner(**player_cfg, map_path=cfg["env"]["map_path"], player_id=i))
    return players


def main():
    env = MicroRTSLLMEnv(agents=get_players(), **cfg["env"])
    
    payoff, trajectory = env.run()

    print(payoff)

    trajectory.to_json("runs/trajectory.json")

    with open("runs/metric.txt", "w") as f:
        f.write(env.metric.to_string())


if __name__ == "__main__":
    main()

