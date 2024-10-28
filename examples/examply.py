from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.base_agent import BlueAgent, RedAgent
from skill_rts.agents.bot_agent import randomAI, coacAI
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = MicroRTSLLMEnv([BlueAgent(), randomAI], record_video=False, theme="black", display=False)

payoffs, trajectory = env.run()

print(payoffs)

with open("traj.txt", "w") as f:
    print(trajectory.to_string(), file=f)
