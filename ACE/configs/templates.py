import yaml


with open("/root/desc/skill-rts/ACE/configs/templates/template_planning.yaml") as f:
    base = yaml.safe_load(f)

INSTRUCTION = base["INSTRUCTION"]
INTRODUCTION = base["INTRODUCTION"]
EXAMPLES = base["EXAMPLES"]
TIPS = base["TIPS"]
OPPONENT = base["OPPONENT"]
STRATEGY = base["STRATEGY"]
START = base["START"]

zero_shot = INSTRUCTION + INTRODUCTION + START

few_shot = INSTRUCTION + INTRODUCTION + EXAMPLES + START

few_shot_w_strategy = INSTRUCTION + INTRODUCTION + EXAMPLES + STRATEGY + START

with open("/root/desc/skill-rts/ACE/configs/templates/template_strategy.yaml") as f:
    strategy = yaml.safe_load(f)
