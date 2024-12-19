import json
import torch
import time
import os
import numpy as np
from ace.strategy import Strategy
from ace.offline.cnn import CNN1D
from ace.agent import Planner
from skill_rts.envs.wrappers import MicroRTSLLMEnv
from omegaconf import OmegaConf
from skill_rts import logger


logger.set_level(logger.DEBUG)

cfg = {
    "model": "Qwen2.5-72B-Instruct",
    "temperature": 0,
    "max_tokens": 8192,
    "prompt": "few-shot-w-strategy",
    "map_name": "basesWorkers8x8"
}


def eval_payoff():
    models_path = "ace/data/payoff/cnn1d.pth"
    runs_dir = "runs/test_payoff"
    model = CNN1D()
    model.load_state_dict(torch.load(models_path, weights_only=True))
    model.eval()
    model.cuda()

    feat_space = Strategy.feat_space()
    
    opponents = os.listdir("ace/data/test")
    for opponent in opponents:
        run_dir = f"{runs_dir}/{opponent}"
        
        # Search for the best response to the opponent
        opponent_strategy = Strategy.load_from_json(f"ace/data/test/{opponent}")
        opponent_feats = np.tile(opponent_strategy.feats, (feat_space.shape[0], 1))
        X = torch.tensor(np.hstack([feat_space, opponent_feats]), dtype=torch.float32, device="cuda")
        X = X.unsqueeze(1)
        with torch.no_grad():
            win_rate = torch.nn.functional.softmax(model(X), dim=1)
        max_idx = win_rate[:, 0].argmax().item()
        response = Strategy.decode(feat_space[max_idx]).strategy

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
        print(f"Opponent {opponent} | Payoffs: {payoffs} | Runtime: {(end_time - start_time) / 60:.2f}min, {env.time}steps")


if __name__ == "__main__":
    # gen_eval_strategies()
    eval_payoff()
