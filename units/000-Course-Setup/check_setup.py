import sys
import gymnasium as gym
import stable_baselines3 as sb3
import huggingface_sb3
import huggingface_hub
from stable_baselines3.common.env_util import make_vec_env

def check_setup():
    print("--- Reinforcement Learning Setup Check ---")
    
    # 1. Check Python Version
    print(f"Python version: {sys.version.split()[0]}")

    # 2. Check Core Library Versions
    print(f"Gymnasium version: {gym.__version__}")
    print(f"Stable Baselines3 version: {sb3.__version__}")
    print(f"Hugging Face SB3 version: {huggingface_sb3.__version__}")
    
    # 3. Verify Box2D (Lunar Lander) Installation
    print("\nTesting Environment: LunarLander-v3...")
    try:
        env = gym.make("LunarLander-v3")
        env.reset()
        print("Success: Box2D environments are correctly installed.")
        env.close()
    except Exception as e:
        print(f"Error loading Box2D: {e}")
        print("Troubleshooting: Try running 'pip install swig' followed by 'pip install gymnasium[box2d]'")

    # 4. Check Hugging Face Hub Authentication
    print("\nChecking Hugging Face Hub...")
    try:
        user = huggingface_hub.whoami()
        print(f"Success: Logged in as: {user['name']}")
    except Exception:
        print("Warning: Not logged in to Hugging Face Hub.")
        print("Troubleshooting: Run 'huggingface-cli login' or use notebook_login() in your script.")

    print("\n--- Setup Check Complete ---")

if __name__ == "__main__":
    check_setup()
