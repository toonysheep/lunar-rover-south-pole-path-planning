"""Load one original dataset, reset, and execute a few fixed actions; no learning."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", choices=("site01", "site06", "site07"), default="site01")
    p.add_argument("--start-time", type=int, default=0)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--data-root", type=Path)
    args = p.parse_args()
    if not 0 <= args.steps <= 10:
        p.error("--steps must be between 0 and 10 for this smoke check")
    from lunar_rover_env import LunarRoverEnv
    env = LunarRoverEnv(site=args.site, start_time=args.start_time, data_root=args.data_root)
    try:
        obs = env.reset()
        print(json.dumps({"observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                          "position": list(map(int, env.agent_pos)), "goal": list(map(int, env.goal_pos))}))
        for _ in range(args.steps):
            obs, reward, done, info = env.step(0)
            print(json.dumps({"step": env.step_count, "action": 0, "reward": float(reward), "done": bool(done)}))
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
