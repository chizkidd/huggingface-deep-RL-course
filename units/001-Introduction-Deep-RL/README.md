# Unit 1: Introduction to Deep Reinforcement Learning

## 1.1 Introduction

Deep Reinforcement Learning (Deep RL) is a subfield of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** to maximize a **reward**.

* **The Goal:** Learn an optimal strategy (Policy) to get the most rewards over time.
* **Hands-on:** In this unit, we train a `LunarLander-v2` agent using the **Stable Baselines3** library.

## 1.2 What is RL?

>Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

* **The Analogy:** Like a child learning to play a video game. They press buttons (actions), see the score increase (reward) or lose a life (punishment), and adjust their behavior accordingly.
* **Reward Hypothesis:** All goals can be described as the ***maximization of the expected return (expected cumulative reward).***
  > That’s why in Reinforcement Learning, ***to have the best behavior,*** we aim to learn to take actions that **maximize the expected cumulative reward.**

## 1.3 The RL Framework

The RL process is a loop that happens at each time step :

1. **State ($S_t$):** The agent receives the current state from the environment.
2. **Action ($A_t$):** The agent takes an action based on that state.
3. **Reward ($R_{t+1}$):** The environment gives a reward.
4. **Next State ($S_{t+1}$):** The environment transitions to a new state.

### Key Concepts:

* **Markov Property:** The agent only needs the current state to decide the next action; the history of past states and actions is irrelevant.
  > _The future is independent of the past given the present."_
  > _The memoryless property of a stochastic process, which means that its future evolution is independent of its history._


* **Observations vs. States:** Observations/States are the information our agent gets from the environment.
  - **State, $s$:** A complete description of the state of the world. No hidden information. (e.g., Chess, complete board view).
  - **Observation, $o$:** A partial description of the state of the world. In a partially observed environment (e.g., Super Mario where you only see the current screen).
    
_Note: In this course, we use the term "state" to denote both state and observation, but we will make the distinction in implementations._


* **Action Space:** The Action space is the set of all possible actions in an environment.
  - **Discrete:** Finite number of possible actions (Up, Down, Left, Right).
  - **Continuous:** Infinite number of possible actions (Steering wheel angle from -180° to 180°).


* **Rewards and Discounting:**

  - The reward is the only feedback for the agent. The cumulative reward at each time step  can be written as: $$G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...$$
  - However, in reality, rewards that come sooner are more likely to happen and more predictable than long-term future rewards. We use **discounting** to account for this.
  - Each reward is discounted by gamma to the exponent of the time step: $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$.
  - This means as the time step increases (the further into the future the reward is), the less it contributes to the immediate decision.

* **The Discount Rate ($\gamma$):** We define a discount rate called **gamma ()**. It must be between 0 and 1 (typically 0.95 to 0.99).
  - **Larger  (closer to 1):** Smaller discount; the agent cares more about long-term rewards.
  - **Smaller  (closer to 0):** Larger discount; the agent cares more about short-term rewards (the "nearest cheese").


## 1.4 Tasks: Episodic vs. Continuing

* **Episodic Tasks:** Have a clear starting and ending point (e.g., a game of Pong).
* **Continuing Tasks:** Tasks that continue forever without a terminal state (e.g., automated stock trading).

## 1.5 The Exploration/Exploitation Trade-off

* **Exploration:** Trying random actions to find more information about the environment (searching for a bigger cheese).
* **Exploitation:** Using known information to maximize rewards (eating the small cheese right in front of you).
* **The Challenge:** Finding the right balance so the agent doesn't get stuck in "local optima."

## 1.6 Two Main Approaches

To find the **Optimal Policy ($\pi^*$)**, we use two main methods:

1. **Policy-Based Methods:** The agent learns the policy function directly. It maps states to actions (or probabilities of actions).
2. **Value-Based Methods:** The agent learns a **Value Function** ($V(s)$ or $Q(s,a)$ ) that maps a state to the expected return. The policy then becomes: "Take the action that leads to the state with the highest value."

## 1.7 The "Deep" in Deep RL

* **Classic RL:** Uses a "lookup table" (Q-Table) to store values for every possible state and action. This doesn't scale to complex games with billions of states.
* **Deep RL:** Uses **Deep Neural Networks** as function approximators to predict the best action or value without needing a table.

## 1.8 Summary

* RL is a loop of State $\rightarrow$ Action $\rightarrow$ Reward $\rightarrow$ Next State.
* The objective is to maximize the **Expected Return** (discounted cumulative reward).
* We use **Discounting ($\gamma$)** because immediate rewards are more certain than distant future rewards.

## 1.9 Glossary

* **Agent:** The AI learner/decision maker.
* **Environment:** The world the agent lives in.
* **Policy ($\pi$):** The "brain" of the agent—the strategy it follows.
* **Value Function:** Tells us how "good" it is for the agent to be in a certain state.
* **Gamma ($\gamma$):** The discount rate (usually between 0.95 and 0.99).

---

## ![](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white) Introduction to Deep Reinforcement Learning | Deep RL Course
This video covers the foundational concepts of the RL loop, the reward hypothesis, markov property, exploration/exploitation tradeofff, policy, and the difference between policy-based and value-based methods mentioned in my notes.<br><br>
*Click the image below to watch the Unit 1 Introduction*

[![Watch the video](https://img.youtube.com/vi/q0BiUn5LiBc/0.jpg)](https://www.youtube.com/watch?v=q0BiUn5LiBc)



