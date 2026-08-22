# ==========================================
# TwinEV v2 - train a single PPO model
# ==========================================
# Used to parallelize the many identical
# trainings required by experiments 1-5.
#   python train_single.py <tag> <env_cls> <save_path> [seed] [weights_key]
# env_cls in {twin, noforecast, nopi}
# ==========================================

import argparse
import json
import sys
import time

from exp_utils import NoForecastEnv, NoPIEnv, TwinEVEnv
from train_agent import train_agent

ENV_CLASSES = {
    "twin": TwinEVEnv,
    "noforecast": NoForecastEnv,
    "nopi": NoPIEnv,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("env_cls", choices=ENV_CLASSES.keys())
    parser.add_argument("save_path")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    weights = None
    if args.weights:
        with open("results/reward_sensitivity/reward_configs.json") as f:
            configs = json.load(f)
        cfg = configs[args.weights]
        weights = {k: v for k, v in cfg.items()
                   if k not in ("description", "lookahead_bonus_weight")}
        print(f"[{args.tag}] reward weights: {cfg}")

    t0 = time.time()
    model, final_reward = train_agent(
        ENV_CLASSES[args.env_cls], args.save_path,
        reward_weights=weights,
        seed=args.seed,
        total_timesteps=args.timesteps,
    )
    dt = time.time() - t0
    print(f"[{args.tag}] DONE in {dt/60:.1f} min | final train reward {final_reward:.2f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
