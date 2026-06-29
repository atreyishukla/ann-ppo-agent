import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from config import DATA_PATH, MODEL_DIR, SHEET_NAME
from greenhouse_env import GreenhouseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--target-temp", type=float, default=22.0)
    parser.add_argument("--comfort-band", type=float, default=1.0)
    parser.add_argument("--episode-len", type=int, default=24)
    parser.add_argument("--curriculum-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    for old_model in [
        f"{MODEL_DIR}/dqn_greenhouse_agent.zip",
        f"{MODEL_DIR}/ppo_greenhouse_agent.zip",
    ]:
        if os.path.exists(old_model):
            os.remove(old_model)
            print(f"Removed old RL model: {old_model}")

    env = GreenhouseEnv(
        DATA_PATH,
        SHEET_NAME,
        target_temp=args.target_temp,
        comfort_band=args.comfort_band,
        episode_len=args.episode_len,
        curriculum_prob=args.curriculum_prob,
    )

    env.save_observation_stats()
    check_env(env, warn=True)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.92,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 128, 64]},
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(f"{MODEL_DIR}/ppo_greenhouse_agent")

    print("Saved PPO 4-state RL agent to models/ppo_greenhouse_agent.zip")
    print("Run: python src/test_controller_cases.py")


if __name__ == "__main__":
    main()