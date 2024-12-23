import json
import time
import os
from ace.offline.payoff_net import PayoffNet
from ace.agent import Planner
from ace.strategy import Strategy
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from omegaconf import OmegaConf
from skill_rts import logger


def eval_model():
    logger.set_level(logger.DEBUG)
    cfg = {
        "model": "Qwen2.5-72B-Instruct",
        "temperature": 0,
        "max_tokens": 8192,
        "prompt": "few-shot-w-strategy",
        "map_name": "basesWorkers8x8"
    }
    runs_dir = "runs/test_payoff"
    model = PayoffNet.load("ace/data/payoff/payoff_net.pth")

    feat_space = Strategy.feat_space()
    
    opponents = os.listdir("ace/data/test")
    for opponent in opponents:
        run_dir = f"{runs_dir}/{opponent}"
        
        # Search for the best response to the opponent
        opponent_strategy = Strategy.load_from_json(f"ace/data/test/{opponent}")
        response, win_rate = model.search_best_response(feat_space, opponent_strategy.feats)
        print(f"Opponent: {opponent} | Win Rate: {win_rate:.2f}", end=" ", flush=True)

        # Run the game
        agent = Planner(player_id=0, strategy=response, **cfg)
        opponent_agent = Planner(player_id=1, strategy=opponent_strategy, **cfg)
        env = MicroRTSLLMEnv([agent, opponent_agent], interval=200, record_video=True, run_dir=run_dir)
        start_time = time.time()
        try:
            payoffs, traj = env.run()
        except Exception as e:
            print(f"Error running game against {opponent}: {e}")
            env.close()
            continue
        end_time = time.time()

        # Save the results
        OmegaConf.save(cfg, f"{run_dir}/config.yaml")
        traj.to_json(f"{run_dir}/traj.json")
        env.metric.to_json(f"{run_dir}/metric.json")
        with open(f"{run_dir}/plans.json", "w") as f:
            json.dump(env.plans, f, indent=4)
        print(f"Payoffs: {payoffs} | Runtime: {(end_time - start_time) / 60:.2f}min, {env.time}steps")


if __name__ == "__main__":
    eval_model()