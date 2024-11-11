import argparse
from omegaconf import OmegaConf


def config(config_path: str = "/root/desc/skill-rts/ace/configs/config.yaml"):

    cfg = OmegaConf.load(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, help='Path to the map file')
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps')
    parser.add_argument('--interval', type=int, help='Interval for call LLM')
    parser.add_argument('--record_video', action='store_true', help='Whether to record video')
    parser.add_argument('--display', action='store_true', help='Whether to display the environment')
    parser.add_argument('--temperature', type=float, help='Temperature for LLM')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens for LLM')

    args = parser.parse_args()

    if args.map_path is not None:
        cfg.env.map_path = args.map_path
    if args.max_steps is not None:
        cfg.env.max_steps = args.max_steps
    if args.interval is not None:
        cfg.env.interval = args.interval
    if args.record_video:
        cfg.env.record_video = True
    if args.display:
        cfg.env.display = True
    if args.temperature is not None:
        cfg.llm.temperature = args.temperature
    if args.max_tokens is not None:
        cfg.llm.max_tokens = args.max_tokens
    
    return OmegaConf.to_container(cfg, resolve=True)

cfg = config()


if __name__ == "__main__":
    print(cfg)
