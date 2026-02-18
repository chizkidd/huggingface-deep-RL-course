![Sanitizer Status](https://github.com/chizkidd/huggingface-deep-RL-course/actions/workflows/clean_notebooks.yml/badge.svg)

# Hugging Face Deep RL Course 

This repository documents my journey through the [Hugging Face Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course/). It contains my progress, notes, implementations, notebooks, environment setups, and the agents I've trained to master various tasks.

--- 
## Goals
- Understand the theoretical foundations of Reinforcement Learning.
- Learn to use libraries like **Stable Baselines3**, **Gymnasium**, and **RL Baselines3 Zoo**.
- Train agents to play games (Lunar Lander, Atari) and solve robotics tasks.
- Share my models on the Hugging Face Hub.

---
## Progress Tracker
| Unit | Topic | Status | Model Hub Link | Notes Link |
| :--- | :--- | :---: | :--- | :--- |
| **0** | Introduction & Setup | ✅ | - | [Notes](units/000-Course-Setup/README.md) |
| **1** | Intro to Deep RL (`LunarLander-v3`) | ✅ | [ppo-LunarLander-v3](https://hf.co/Chiz/ppo-LunarLander-v3) | [Notes](units/001-Introduction-Deep-RL/README.md) |
| **1b** | **Bonus:** `Huggy-the-Dog` | ✅ |  [ppo-Huggy](https://hf.co/Chiz/ppo-Huggy) | [Notes](units/001b-Huggy-the-dog/README.md) |
| **2** | Q-Learning (`FrozenLake-v1` & `Taxi-v3`) | ✅ | [q-FrozenLake-v1-4x4-noSlippery](https://hf.co/Chiz/q-FrozenLake-v1-4x4-noSlippery),<br> [q-Taxi-v3](https://hf.co/Chiz/q-Taxi-v3) | [Notes](units/002-Q-Learning/README.md) |
| **3** | Deep Q-Learning (`Atari Space Invaders`) | ✅ | [dqn-SpaceInvadersNoFrameskip-v4](https://hf.co/Chiz/dqn-SpaceInvadersNoFrameskip-v4) | [Notes](units/003-Deep-Q-Learning/README.md) |
| **3b** | **Bonus:** `Optuna` Hyperparameter Tuning | 🏗️ | [Link] | [Notes](units/003b-Optuna/README.md) |
| **4** | Policy Gradients (`CartPole-v1`) | ✅ | [Reinforce-CartPole-v2](https://hf.co/Chiz/Reinforce-CartPole-v2),<br> [Pixelcopter-PLE-v0](https://hf.co/Chiz/Reinforce-Pixelcopter-PLE-v0) | [Notes](units/004-Policy-Gradients/README.md) |
| **5** | Unity ML-Agents | ⏳ | [Link] | [Notes](units/005-Unity-ML-Agents/README.md) |
| **6** | Actor-Critic Methods (Robotics) | ✅ | [a2c-PandaReachDense-v3](https://hf.co/Chiz/a2c-PandaReachDense-v3) | [Notes](units/006-Actor-Critic/README.md) |
| **7** | Multi-Agent RL (Soccer) | ✅ | [poca-SoccerTwos](https://hf.coChiz/poca-SoccerTwos) | [Notes](units/007-Multi-Agent-RL/README.md) |
| **8** | PPO Part 1: Theory & Implementation | ✅ | [ppo-LunarLander-v3-02](https://hf.co/Chiz/ppo-LunarLander-v3-02) | [Notes](units/008-PPO/README.md) |
| **8b** | PPO Part 2: `VizDoom` | ⬜ | [Link] | [Notes](units/008b-PPO-2/README.md) |
| **9** | **Bonus:** Advanced Topics in RL | ⬜ | [Link] | [Notes](units/009-Advanced-Topics-RL/README.md) |
| **10** | **Bonus:** Imitation Learning (`Godot`) | ⬜ | [Link] | [Notes](units/010-Imitation-Learning/README.md) |


Currently ongoing ...

---
## Resources

* [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) — The main course curriculum.
* [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/RLbook2020.pdf) — The "Bible" of RL; essential for deep theoretical understanding.

---
## Local Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/chizkidd/huggingface-deep-RL-course.git
   cd huggingface-deep-RL-course
