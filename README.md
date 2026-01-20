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
| **1** | Intro to Deep RL (`LunarLander-v3`) | ✅ | [Chiz/ppo-LunarLander-v3](https://hf.co/Chiz/ppo-LunarLander-v3) | [Notes](units/001-Introduction-Deep-RL/README.md) |
| **1b** | **Bonus:** `Huggy-the-Dog` | ✅ |  [Chiz/ppo-Huggy](https://hf.co/Chiz/ppo-Huggy) | [Notes](units/001b-Huggy-the-dog/README.md) |
| **2** | Q-Learning (`FrozenLake-v1` & `Taxi-v3`) | 🏗️ | [Link] | [Notes](units/002-Q-Learning/README.md) |
| **3** | Deep Q-Learning (`Atari Space Invaders`) | ⬜ | [Link] | [Notes](#) |
| **3b** | **Bonus:** `Optuna` Hyperparameter Tuning | ⬜ | [Link] | [Notes](#) |
| **4** | Policy Gradients (`CartPole-v1`) | ⬜ | [Link] | [Notes](#) |
| **5** | Unity ML-Agents | ⬜ | [Link] | [Notes](#) |
| **6** | Actor-Critic Methods (Robotics) | ⬜ | [Link] | [Notes](#) |
| **7** | Multi-Agent RL (Soccer) | ⬜ | [Link] | [Notes](#) |
| **8** | PPO Part 1: Theory & Implementation | ⬜ | [Link] | [Notes](#) |
| **8b** | PPO Part 2: `VizDoom` | ⬜ | [Link] | [Notes](#) |
| **9** | **Bonus:** Advanced Topics in RL | ⬜ | [Link] | [Notes](#) |
| **10** | **Bonus:** Imitation Learning (`Godot`) | ⬜ | [Link] | [Notes](#) |

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
