"""Export the PPO constructor and evaluation settings used by the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT = Path(__file__).parent / "outputs" / "ppo_hyperparameters.csv"


def main() -> None:
    rows = [
        ("learning_rate", "3e-4", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("n_steps", "2048", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("batch_size", "256", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("n_epochs", "10", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("gamma", "0.99", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("gae_lambda", "0.95", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("clip_range", "0.2", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("ent_coef", "0.01", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("vf_coef", "0.5", "explicit", "train_controller.py:489; exp_2x2_run.py:250"),
        ("net_arch", "[256, 256, 128]", "explicit", "policy_kwargs at train_controller.py:489"),
        ("activation_fn", "Tanh", "library default", "Stable-Baselines3 MlpPolicy default"),
        ("deterministic", "True", "explicit", "train_controller.py:529; exp_2x2_run.py:277"),
    ]
    result = pd.DataFrame(
        rows, columns=["parameter", "value", "source_classification", "source"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
