# Unit 7: Multi-Agent Reinforcement Learning (MARL)

## Table of Contents
1. [Introduction](#71-introduction)
2. [Introduction to MARL](#72-introduction-to-marl)
3. [The Multi-Agent Setting](#73-the-multi-agent-setting)
    * [Taxonomy of MA Settings](#taxonomy-of-ma-settings)
    * [The Challenges of MARL](#the-challenges-of-marl)
4. [Self-Play](#74-self-play)
5. [Glossary](#75-glossary)

---

## 7.1 Introduction
In previous units, we focused on **Single-Agent Reinforcement Learning**, where one agent interacts with an environment to maximize its own reward. However, many real-world scenarios—like robotics, autonomous driving, and multiplayer games—involve multiple agents interacting in the same space.

This unit explores how agents learn when their success depends not only on their own actions but also on the actions of others.

---

## 7.2 Introduction to MARL
**Multi-Agent Reinforcement Learning (MARL)** is the study of multiple agents that coexist in a shared environment. 

### Key Characteristics:
* **Interaction:** Agents can cooperate, compete, or exist in a mix of both.
* **Dynamic Environment:** From the perspective of one agent, the environment is no longer stationary because other agents are learning and changing their behavior simultaneously.

---

## 7.3 The Multi-Agent Setting

### Taxonomy of MA Settings
Depending on the rewards, multi-agent interactions generally fall into three categories:

1.  **Cooperative (Common Interest):** All agents work together to achieve a shared goal.
    * *Equation:* $r_1 = r_2 = ... = r_n$
2.  **Competitive (Zero-Sum):** One agent's gain is another's loss. Common in board games and duels.
    * *Equation:* $\sum r_i = 0$
3.  **Mixed (General Sum):** Agents have their own reward functions, which may align or conflict depending on the situation (e.g., traffic navigation).

### The Challenges of MARL
MARL is significantly more difficult than single-agent RL due to:

#### 1. Non-Stationarity
In single-agent RL, the environment transition probability $P(s'|s, a)$ is constant. In MARL, the transition depends on the joint action $\mathbf{a} = (a_1, ..., a_n)$:
$$P(s' | s, a_1, ..., a_n)$$
Because other agents are also updating their policies $\pi_i$, the environment appears to change constantly from the perspective of Agent 1. This violates the Markov property required for standard RL algorithms.

#### 2. Curse of Dimensionality
As the number of agents increases, the state-action space grows exponentially. If each of $N$ agents has $M$ possible actions, the joint action space is $M^N$.

#### 3. Credit Assignment
In a cooperative setting, if the group receives a reward, it is difficult to determine which specific agent’s action contributed most to that success.

---

## 7.4 Self-Play
**Self-Play** is a powerful training technique where an agent learns by playing against versions of itself.

### How it works:
1.  The agent starts with a random policy.
2.  It plays against its current version or previous versions stored in a "buffer" of opponents.
3.  As the agent improves, its opponent (itself) also becomes stronger, creating a natural "curriculum" of increasing difficulty.

### Advantages:
* **No Human Data Needed:** The agent discovers strategies from scratch (e.g., AlphaGo Zero).
* **Infinite Opponents:** You always have an opponent at exactly your skill level.

### The Problem of "Cycles":
A risk in self-play is that an agent might learn to beat version $A$, then version $B$ beats version $A$, but version $A$ can beat version $C$. This rock-paper-scissors cycle can prevent the agent from reaching a truly robust strategy.

---

## 7.5 Glossary
* **Joint Action:** The vector containing the individual actions of every agent in the environment at a specific time step.
* **Non-Stationarity:** The phenomenon where an agent's environment changes because other agents in the environment are also learning and changing their behavior.
* **Markov Game (Stochastic Game):** The multi-agent extension of a Markov Decision Process (MDP).
* **Zero-Sum Game:** A setting where the total sum of rewards for all agents is always zero; one agent can only win if another loses.
* **Centralized Training, Decentralized Execution (CTDE):** A common MARL framework where agents have access to extra information during training but must act based only on local observations during execution.
* **Nash Equilibrium:** A state in a game where no agent can increase its expected reward by changing its strategy, provided all other agents keep theirs unchanged.
* **Self-Play:** A training method where an agent improves by competing against previous iterations of its own policy.