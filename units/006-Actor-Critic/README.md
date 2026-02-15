# Unit 6: Actor-Critic Methods

## Table of Contents
1. [Introduction](#61-introduction)
    * [The Architecture](#the-architecture)
2. [The Problem of High Variance](#62-the-problem-of-high-variance)
    * [Why Variance is a Challenge](#why-variance-is-a-challenge)
    * [The Solution: Baselines](#the-solution-baselines)
3. [Advantage Actor-Critic (A2C)](#63-advantage-actor-critic-a2c)
    * [The Actor and The Critic](#the-actor-and-the-critic)
    * [The Advantage Function](#the-advantage-function)
        * [The Math](#the-math)
    * [Simplified Derivation](#simplified-derivation)
        * [1. Actor Update/Loss (Policy Gradient)](#1-actor-updateloss-policy-gradient)
        * [2. Critic Update/Loss (Value Regression)](#2-critic-updateloss-value-regression)
        * [3. Entropy Bonus](#3-entropy-bonus)
    * [A2C vs. A3C](#a2c-vs-a3c)
4. [Glossary](#64-glossary)

---

## 6.1 Introduction
In previous units, we explored two distinct paths: 
* **Value-Based (DQN):** Learning a $Q(s, a)$ to find the best action.
* **Policy-Based (REINFORCE):** Learning a $\pi(s)$ to maximize expected returns. 
 
**Actor-Critic** methods are a hybrid architecture that combine the strengths of both:
* **The Actor:** A Policy-Based component that decides which action to take.
* **The Critic:** A Value-Based component that evaluates those actions by estimating the value function.

Instead of waiting for the end of an episode to see if a series of actions was good, the Actor-Critic allows the agent to learn at every step by using the Critic's feedback to guide the Actor's updates.

### The Architecture
In a neural network implementation, the Actor and Critic often share the first few layers (the "backbone" or "encoder") and then split into two separate heads:

1.  **Policy Head (Actor):** Outputs a probability distribution over actions (Softmax for discrete, Gaussian for continuous).
2.  **Value Head (Critic):** Outputs a single scalar representing $V(s)$.

---

## 6.2 The Problem of High Variance
In pure Policy Gradient methods like REINFORCE, we use the Monte Carlo return $G_t$ to update the policy:
$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

### Why Variance is a Challenge
The return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$ is a single sample of a stochastic process. It depends on:
1.  **Action Randomness:** The agent might take different actions in the same state.
2.  **State Randomness:** The environment transitions might be stochastic.
This creates "noisy" updates that make the policy unstab

$G_t$ represents the total cumulative reward from time $t$ until the end of the episode. Because this return depends on a long sequence of stochastic actions and environment transitions, the value of $G_t$ can vary wildly between episodes.
* **Instability:** High variance in the gradient leads to noisy updates, which can cause the policy to oscillate or diverge.
* **Slow Convergence:** We need a massive amount of samples to "average out" the noise.

### The Solution: Baselines
To reduce this variance without introducing bias, we subtract a **Baseline** $b(s)$ from the return. The most effective baseline is the **Value Function $V(s)$**, which represents the average expected return. By focusing only on how much *better* or *worse* an action is compared to the average, we stabilize the learning signal.

We can reduce variance by subtracting a **Baseline** $b(s)$ that does not depend on the action:
$$\nabla_\theta J(\theta) = E_{\pi} [\nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))]$$
The best baseline is the **Value Function $V(s_t)$**. This is exactly what the Critic provides.

---

## 6.3 Advantage Actor-Critic (A2C)

### The Actor and The Critic
In an A2C implementation, the model usually consists of a shared neural network backbone with two "heads":
1.  **Policy Head (Actor):** Outputs action probabilities $\pi(a|s)$.
2.  **Value Head (Critic):** Outputs a scalar value $V(s)$.

The Critic uses **Bootstrapping**—updating its estimate of $V(s)$ based on the reward $r$ and its own estimate of the next state $V(s')$.

### The Advantage Function
To update the Actor, we use the **Advantage Function** $A(s, a)$. It measures how much better an action is than the average action in that state.

### The Math:
The core of A2C is the **Advantage Function** $A(s, a)$. It quantifies the relative benefit of a specific action:
$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$


We don't want to learn both $Q$ and $V$, so we use the **TD Error** ($\delta$) as an estimator for the Advantage. Since $Q(s_t, a_t)$ can be approximated using the Temporal Difference (TD) target, we rewrite the Advantage as the **TD Error ($\delta$)**:
$$A(s_t, a_t) \approx \delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

* If $\delta_t > 0$: The action was better than expected (increase probability).
* If $\delta_t < 0$: The action was worse than expected (decrease probability).

### Simplified Derivation
The training objective is to minimize a combined loss function:
$$L_{total} = L_{actor} + L_{critic} + L_{entropy}$$

#### 1. Actor Update/Loss (Policy Gradient)
We replace the high-variance $G_t$ with the Advantage. Using the Advantage estimate:
$$L_{actor} = -\log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t)$$

#### 2. Critic Update/Loss (Value Regression)
The Critic is needed to predict the true value of the state. The Critic minimizes the Mean Squared Error (MSE) between its prediction and the TD target:
$$L_{critic} = \frac{1}{2} ( (r_{t+1} + \gamma V(s_{t+1})) - V(s_t) )^2$$

#### 3. Entropy Bonus
To ensure the Actor continues to explore, we add an **Entropy** term. This prevents the policy from becoming deterministic too early by rewarding "uncertainty" in the action distribution.

To prevent the policy from collapsing into a single deterministic action too early (improving exploration), we add an entropy term:
$$L_{entropy} = -\sum \pi_\theta(a|s) \log \pi_\theta(a|s)$$
High entropy means the agent is exploring; low entropy means it's confident.

## A2C vs. A3C
* **A3C (Asynchronous Advantage Actor-Critic):** Multiple independent agents interact with their own copies of the environment. They update a global master model asynchronously.
* **A2C (Advantage Actor-Critic):** A synchronous version. It waits for all agents to finish their segment of experience, averages their gradients, and then updates the global model once. A2C is often more efficient on GPUs.

---

## 6.4 Glossary
* **Actor:** The component that learns the policy $\pi(s, a)$ (mapping states to actions).
* **Critic:** The component that learns the value function $V(s)$ (mapping states to expected returns).
* **Advantage ($A$):** The measure of how much better an action is than the average action in a state ($Q - V$).
* **Variance:** The "noise" in the reward signal that makes training unstable.
* **TD Target:** The "true" value we aim for: $r + \gamma V(s')$.
* **Temporal Difference (TD) Error:** The difference between the estimated value of the current state and the actual reward plus the estimated value of the next state. The difference between the TD Target ($r + \gamma V(s')$) and the current value estimate $V(s)$.
* **Bootstrapping:** The process of updating an estimate using other estimates (characteristic of the Critic/TD learning), for example, using $V(s_{t+1})$ to update $V(s_t)$
* **A2C ([Synchronous] Advantage Actor-Critic):** A version where the master model waits for all parallel workers to finish their segments before performing a single gradient update.
* **A3C (Asynchronous Advantage Actor-Critic):** An architecture where multiple independent agents interact with their own copies of the environment and update a global master model asynchronously, allowing for faster and more diverse training.
* **A2C vs A3C:** A2C is synchronous (waits for all workers), while A3C is asynchronous (workers update the global model independently).
* **Entropy:** A measure of randomness in the policy; used as a bonus to encourage exploration.
* **Shared Parameters:** A common deep learning practice where the Actor and Critic share the same feature extraction layers.