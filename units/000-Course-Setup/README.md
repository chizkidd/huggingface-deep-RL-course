# Unit 0: Setup and Workspace Foundations 

This unit covers the essential environment setup and authentication required to train agents and share them with the community.

## Environment Setup

To ensure consistency across all units, I am using a virtual environment with the core libraries installed.

### Core Dependencies

* **Gymnasium:** The standard API for reinforcement learning environments.
* **Stable Baselines3:** Reliable implementations of RL algorithms (PPO, DQN, etc.).
* **Hugging Face SB3:** Integration for pushing/loading models to the Hub.

### Initialization Command

```bash
# Installing the "essentials" for the course
pip install gymnasium[box2d] stable-baselines3[extra] huggingface_sb3 huggingface_hub

```

## Hugging Face Hub Integration

To participate in the leaderboards and share models, authentication is required.

1. **Account:** [Chiz](https://huggingface.co/Chiz)
2. **Login:** Use `huggingface-cli login` or the code snippet below in notebooks:
```python
from huggingface_hub import notebook_login
notebook_login()

```


## Learning Path & Certification

* **Goal:** Complete 80% of the course assignments.
* **Method:** Practical notebooks in Google Colab + Local development for theory.
* **Status:** Hugging Face account created and Discord joined for community support.


## Workspace Verification

To ensure that the foundation is solid before moving into complex algorithms, I use a custom verification script. This script checks library versions, validates **Box2D** installation (required for `LunarLander`), and confirms Hugging Face authentication status.

### Setup Check Script

The script `check_setup.py` is located in this directory. It provides a quick way to diagnose environment issues.

**Run the verification:**

```bash
python units/000-Course-Setup/check_setup.py

```

### What it validates:

* **Versions:** Displays current versions of `gymnasium`, `stable-baselines3`, and `huggingface_hub`.
* **Box2D Check:** Attempts to initialize `LunarLander-v3` to ensure physics engines are properly compiled.
* **HF Auth:** Checks if a valid Hugging Face token is detected in the local cache.

---

