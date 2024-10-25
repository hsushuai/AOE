from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.base_agent import Agent
from skill_rts.agents.bot_agent import randomAI

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = MicroRTSLLMEnv([Agent(), randomAI])
payoffs = env.run()

print(payoffs)
