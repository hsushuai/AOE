from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.base_agent import BlueAgent, RedAgent
from skill_rts.agents.bot_agent import randomAI

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = MicroRTSLLMEnv([BlueAgent(), randomAI], record_video=True, theme="black")
payoffs = env.run()

print(payoffs)
