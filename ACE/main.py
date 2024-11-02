from skill_rts.envs import MicroRTSLLMEnv
from skill_rts.agents import bot_agent
from ACE.llm_agent import LLMAgent
from ACE.configs.config import cfg
from skill_rts import logger

logger.set_level(logger.DEBUG)


def get_players():
    players_cfg = cfg["players"]
    players = []
    for i, player_cfg in enumerate(players_cfg):
        if player_cfg["model"] in bot_agent.ALL_AIS:
            players.append(bot_agent.get_agent(player_cfg["model"]))
        else:
            players.append(LLMAgent(**player_cfg, map_path=cfg["env"]["map_path"], player_id=i))
    return players


def main():
    env = MicroRTSLLMEnv(agents=get_players(), **cfg["env"])
    
    payoff, trajectory = env.run()

    print(payoff)

    with open("runs/traj.txt", "w") as f:
        print(trajectory.to_string(), file=f)

    with open("runs/metric.json", "w") as f:
        env.metric.to_json(f)


if __name__ == "__main__":
    # main()
    feature_value = "Some Feature"  # 这里替换为你想要的 feature 值

    barrack_feature = """**Barracks Feature** determines when to build barracks. You can construct barracks when resources are greater than or equal to a specific threshold (N), or not build them at all (False).
  - Parameter: Timing of construction (resource quantity)
  - Feature Space: {{resource >= N, False}}
  - Value: {feature}"""

    formatted_barrack_feature = barrack_feature.format(feature=feature_value)

    print(formatted_barrack_feature)

