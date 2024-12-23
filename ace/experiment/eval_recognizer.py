import numpy as np
from ace.agent import Recognizer
from ace.strategy import Strategy
from ace.traj_feat import TrajectoryFeature
from skill_rts.game.trajectory import Trajectory
from skill_rts import logger


def eval_recognizer():
    logger.set_level(logger.DEBUG)
    cfg = {
        "model": "Qwen2.5-72B-Instruct",
        "temperature": 0,
        "max_tokens": 8192
    }
    log_file = open("runs/eval_recognizer.log", "w")
    logger.set_stream(log_file)

    recognizer = Recognizer(**cfg)
    online_runs_dir = "runs/online_runs"

    correct = 0
    for i in range(1, 51):
        logger.info(f"{'-' * 20} Strategy {i} {'-' * 20}")
        traj = Trajectory.load(f"{online_runs_dir}/strategy_{i}/run_0/traj.json")
        traj_feat = TrajectoryFeature(traj)
        recog_strat = recognizer.step(traj_feat.to_string())
        true_strat = Strategy.load_from_json(f"ace/data/strategies/strategy_{i}.json")
        dist = np.linalg.norm(recog_strat.feats - true_strat.feats)
        logger.info(f"GROUND TRUTH: {true_strat.strategy}")
        logger.info(f"GROUND TRUTH FEATS: {true_strat.feats}\n")
        logger.info(f"RECOGNIZED: {recog_strat.strategy}")
        logger.info(f"RECOGNIZED FEATS: {recog_strat.feats}\n")
        logger.info(f"\nTRAJ_ABS: {traj_feat.to_string()}")
        if recog_strat == true_strat:
            correct += 1
            print("+", end=" ", flush=True)
        else: 
            print(f"{dist:.2f}", end=" ", flush=True)
    
    logger.info(f"\nRecognizer Accuracy: {correct / 50}")
    log_file.close()

if __name__ == "__main__":
    eval_recognizer()