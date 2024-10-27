from skill_rts.envs.wrappers import MicroRTSLLMEnv
from skill_rts.agents.base_agent import BlueAgent, RedAgent
from skill_rts.agents.bot_agent import randomAI, coacAI
from skill_rts.game.game_state import GameState
import json
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = MicroRTSLLMEnv([randomAI, randomAI], record_video=False, theme="black")
payoffs = env.run()
# game_state = env.env.get_game_state()
# trace = str(env.env.vec_client.getTrace())
# trace = json.loads(trace)

# with open("raw_trajectory.json", "w") as f:
#     json.dump(trace, f, indent=4)

# print(str(game_state))

# gs = GameState(raw_entry=trace["entries"][5])
print(payoffs)
