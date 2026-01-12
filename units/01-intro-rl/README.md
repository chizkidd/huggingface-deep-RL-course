# Unit 1: Introduction to Deep Reinforcement Learning

## 1.1 Introduction

Deep Reinforcement Learning (Deep RL) is a subfield of machine learning where an **agent** learns to make decisions by performing **actions** in an **environment** to maximize a **reward**.

* **The Goal:** Learn an optimal strategy (Policy) to get the most rewards over time.
* **Hands-on:** In this unit, we train a `LunarLander-v2` agent using the **Stable Baselines3** library.

## 1.2 What is RL?

Reinforcement Learning is a framework for solving control tasks by trial and error.

* **The Analogy:** Like a child learning to play a video game. They press buttons (actions), see the score increase (reward) or lose a life (punishment), and adjust their behavior accordingly.
* **Reward Hypothesis:** All goals can be described as the maximization of the expected cumulative reward.

## 1.3 The RL Framework

The RL process is a loop that happens at each time step :

1. **State ():** The agent receives the current state from the environment.
2. **Action ():** The agent takes an action based on that state.
3. **Reward ():** The environment gives a reward.
4. **Next State ():** The environment transitions to a new state.

### Key Concepts:

* **Markov Property:** The agent only needs the current state to decide the next action; the history of past states is irrelevant.
* **Observations vs. States:** - **State:** A complete description of the world (e.g., Chess).
* **Observation:** A partial description (e.g., Super Mario where you only see the current screen).


* **Action Space:** - **Discrete:** Finite number of actions (Up, Down, Left, Right).
* **Continuous:** Infinite possibilities (Steering wheel angle from -180° to 180°).



## 1.4 Tasks: Episodic vs. Continuing

* **Episodic Tasks:** Have a clear starting and ending point (e.g., a game of Pong).
* **Continuing Tasks:** Tasks that continue forever without a terminal state (e.g., automated stock trading).

## 1.5 The Exploration/Exploitation Trade-off

* **Exploration:** Trying random actions to find more information about the environment (searching for a bigger cheese).
* **Exploitation:** Using known information to maximize rewards (eating the small cheese right in front of you).
* **The Challenge:** Finding the right balance so the agent doesn't get stuck in "local optima."

## 1.6 Two Main Approaches

To find the **Optimal Policy ()**, we use two main methods:

1. **Policy-Based Methods:** The agent learns the policy function directly. It maps states to actions (or probabilities of actions).
2. **Value-Based Methods:** The agent learns a **Value Function** ( or ) that maps a state to the expected return. The policy then becomes: "Take the action that leads to the state with the highest value."

## 1.7 The "Deep" in Deep RL

* **Classic RL:** Uses a "lookup table" (Q-Table) to store values for every possible state and action. This doesn't scale to complex games with billions of states.
* **Deep RL:** Uses **Deep Neural Networks** as function approximators to predict the best action or value without needing a table.

## 1.8 Summary

* RL is a loop of State  Action  Reward  Next State.
* The objective is to maximize the **Expected Return** (discounted cumulative reward).
* We use **Discounting ()** because immediate rewards are more certain than distant future rewards.

## 1.9 Glossary

* **Agent:** The AI learner/decision maker.
* **Environment:** The world the agent lives in.
* **Policy ():** The "brain" of the agent—the strategy it follows.
* **Value Function:** Tells us how "good" it is for the agent to be in a certain state.
* **Gamma ():** The discount rate (usually between 0.95 and 0.99).

---

## ![](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white) Introduction to Deep Reinforcement Learning | Deep RL Course

*Click the image above to watch the Unit 1 Introduction*

[![Watch the video](https://img.youtube.com/vi/q0BiUn5LiBc/0.jpg)](https://www.youtube.com/watch?v=q0BiUn5LiBc)


This video covers the foundational concepts of the RL loop, the reward hypothesis, and the difference between policy-based and value-based methods mentioned in your notes.
