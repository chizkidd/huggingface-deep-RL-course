# Unit 2: Introduction to Q-Learning

This unit explores the foundations of value-based reinforcement learning. While Unit 1 introduced Deep RL using neural networks, Unit 2 focuses on tabular methods where we learn to estimate the value of states and actions manually using the Bellman Equation.

## 2.1 Introduction

In this unit, you move from Policy-Based methods (where we learn a policy function directly) to **Value-Based methods**. The core objective is to create a **Q-Table**, a cheat sheet that tells the agent the maximum expected future reward for every possible action in every possible state.

## 2.2 What is Value-Based RL?

In Value-Based Reinforcement Learning, the agent learns a **Value Function** that maps a state (or state-action pair) to a value.

* **The Goal:** Find the optimal value function $V^*(s)$ or $Q^*(s, a)$.
* **The Logic:** If we know the value of every state, the optimal policy $\pi^*$ is simply to always take the action that leads to the state with the highest value.

## 2.3 Two Types of Value-Based Methods

1. **State-Value Function $V(s)$:** Calculates the expected return if the agent starts in state $s$ and follows a policy thereafter.
* $V_{\pi}(s) = E_{\pi} [G_t | S_t = s]$
* $V(s) = \mathbb{E}[ R_{t+1} + \gamma V(s_{t+1}) ]$


2. **Action-Value Function $Q(s, a)$:** Calculates the expected return if the agent is in state $s$, takes action $a$, and then follows the policy.
* This is what we use in Q-Learning.



## 2.4 The Bellman Equation

The Bellman Equation is the mathematical foundation of RL. It simplifies the calculation of the value function by breaking the expected return into two parts: the immediate reward plus the discounted value of the next state.
$$V(s) = R + \gamma V(s')$$
$$V(s) = \mathbb{E}[ R_{t+1} + \gamma V(s_{t+1}) ]$$

Where:

* $R$: Immediate reward.
* $\gamma$: Discount factor (importance of future rewards).
* $V(s')$: Value of the next state.

## 2.5 Monte Carlo (MC) vs. Temporal Difference (TD)

These are the two ways we update our value functions:

| Feature | Monte Carlo (MC) | Temporal Difference (TD) |
| --- | --- | --- |
| **Learning** | At the end of the episode. | At every time step (Online). |
| **Requirement** | Needs complete episodes. | Works with incomplete episodes. |
| **Update Base** | Actual total return  $G_t$. | Estimated return $R_{t+1} + \gamma Q(s', a')$. |
| **Bias/Variance** | High variance, zero bias. | Low variance, some bias. |

***<u>Note:</u> Q-Learning uses Temporal Difference (TD) learning.***

## 2.6 Mid-way Recap

* We want to find the optimal policy by finding the optimal value function.
* The value function represents the expected future discounted rewards.
* We use the Bellman Equation to define the value of a state as the sum of immediate reward + discounted future values.
* TD learning allows us to update our estimates at every step without waiting for the episode to end.

## 2.7 Q-Learning: The Algorithm

Q-Learning is an **off-policy** value-based TD algorithm. It uses a **Q-Table** to store the Q-values for all state-action pairs $(s, a)$.

**The Q-Learning Update Rule:**
$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

* **$\alpha$ (Learning Rate):** How much we update our value.
* **$\gamma$ (Discount Factor):** How much we care about future rewards.
* **$\max_{a'} Q(s', a')$:** The greedy estimate of the best future value.

**Off-Policy Definition:** Q-Learning is off-policy because the "acting policy" (how the agent moves, usually $\epsilon$-greedy) is different from the "learning policy" (which assumes the agent will take the absolute best action in the next step).

## 2.8 Q-Learning Example

Imagine a grid world (`FrozenLake`):

1. **Initialize:** Start with a table of zeros.
2. **Choose Action:** Use $\epsilon$-greedy (sometimes explore randomly, sometimes pick the best known action).
3. **Perform Action:** Move, get reward $R$, and see the next state $s'$.
4. **Update:** Use the formula to update the specific cell in the Q-Table.
5. **Repeat:** Do this for thousands of episodes until the table converges.

## 2.9 Q-Learning Recap

* **Q-Table:** A matrix where rows are states and columns are actions.
* **Exploration vs. Exploitation:** We use $\epsilon$-greedy to ensure the agent doesn't get stuck in local optima.
* **Target:** $R + \gamma \max Q(s', a')$.
* **TD Error:** The difference between the target and the current $Q(s, a)$.

## 2.10 Glossary

* **Value-based:** Methods that find the optimal policy by learning the value of states.
* **Q-Value:** The "Quality" or expected reward of taking a specific action in a specific state.
* **Temporal Difference:** Updating an estimate based on another estimate.
* **$\epsilon$-greedy:** A policy that picks a random action with probability $\epsilon$ and the best action with probability $1-\epsilon$.
* **Convergence:** When the Q-Table values stop changing significantly, meaning the agent has found the optimal strategy.
