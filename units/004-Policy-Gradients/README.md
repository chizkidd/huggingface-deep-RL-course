# Unit 4: Policy Gradients - Study Notes

## Table of Contents
1. [Introduction](#41-introduction)
2. [What are Policy-Based Methods?](#42-what-are-policy-based-methods)
3. [Advantages and Disadvantages](#43-advantages-and-disadvantages)
4. [Policy Gradient (The Objective)](#44-policy-gradient-the-objective)
5. [The Policy Gradient Theorem](#45-the-policy-gradient-theorem)
    * [Simplified Derivation](#simplified-derivation)
    * [The REINFORCE Update](#the-reinforce-update)
6. [Glossary](#46-glossary)

---

This unit shifts from **Value-Based** methods (like DQN) to **Policy-Based** methods. Instead of learning which actions are "valuable" and then picking the best one, we learn the policy directly.

---

## 4.1 Introduction

In previous units, we used **Value-Based** methods where the policy $\pi$ was implicit (e.g., $\epsilon$-greedy). We learned a Q-function and acted greedily.
In **Policy-Based** methods, we optimize the policy $\pi$ directly. We parameterize the policy with weights $\theta$ (a neural network) and move $\theta$ in the direction that maximizes the expected return.

---

## 4.2 What are Policy-Based Methods?

In policy-based reinforcement learning, we aim to find the optimal parameters $\theta$ for a policy $\pi_\theta(a|s)$ that maximizes the **Objective Function** $J(\theta)$.

### The Difference:

* **Value-Based:** $V(s)$ or $Q(s, a) \rightarrow$ Indirect Policy.
* **Policy-Based:** $\pi_\theta(a|s) \rightarrow$ Direct Policy.
* **Policy Gradient:** A subclass of policy-based methods that uses **Gradient Ascent** to update $\theta$.

---

## 4.3 Advantages and Disadvantages

### Advantages:

1. **Continuous Action Spaces:** Unlike DQN, which requires a discrete output for every possible action, policy gradients can output parameters of a distribution (like mean $\mu$ and standard deviation $\sigma$), making them ideal for robotics.
2. **Stochastic Policies:** They can learn the optimal probability of actions. This solves **Perceptual Aliasing** (when two different states look the same but require different actions).
3. **Better Convergence:** Policy changes are smoother because the action probabilities change continuously, unlike in value-based methods where a small change in the Q-value can drastically flip the greedy action.

### Disadvantages:

1. **Local Optima:** They are prone to getting stuck in local maxima rather than finding the global optimum.
2. **High Variance:** Gradient estimates can be very noisy, leading to slow or unstable learning/training process.
3. **Sample Inefficiency:** Typically requires more data than value-based methods like DQN or SAC.

---

## 4.4 Policy Gradient (The Objective)

Our goal is to find $\theta$ that maximizes the expected return $J(\theta)$:

$$J(\theta) = E_{\tau \sim \pi_\theta} [R(\tau)]$$

Where $\tau$ is a trajectory $(s_0, a_0, r_1, s_1, ...)$ and $R(\tau)$ is the total reward of that trajectory.
Since we want to maximize this, we use **Gradient Ascent**:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

---

## 4.5 The Policy Gradient Theorem

We cannot calculate the gradient of $J(\theta)$, $\nabla_\theta J(\theta)$, directly because the environment's transition dynamics are **unknown and non-differentiable.** The **Policy Gradient Theorem** provides a way to compute the gradient without knowing the environment's transitions. It provides an analytic expression for the gradient that does not involve the derivative of the state-transition probabilities.

### Simplified Derivation:

1. **The Objective:** $J(\theta) = \sum_{\tau} P(\tau; \theta) R(\tau)$
2. **The Gradient:** $\nabla_\theta J(\theta) = \nabla_\theta \sum_{\tau} P(\tau; \theta) R(\tau) = \sum_{\tau} \nabla_\theta P(\tau; \theta) R(\tau)$
3. **The Log-Derivative Trick:** Since $\nabla_\theta \log P = \frac{\nabla_\theta P}{P}$, we can replace $\nabla_\theta P$ with $P \nabla_\theta \log P$.
      - We use the identity $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$.
4. **The Expectation:**

<center>$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$$</center>
5. **Final Formula:** 

<center>$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau) \right]$$</center>


### The "Reinforce" Update:

In practice, we use a sample-based estimate (Monte Carlo). For each step $t$:

$$\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

By expanding $P(\tau; \theta)$ and removing terms that don't depend on $\theta$, we get the practical update rule:

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

Where $G_t$ is the **Return-to-go** (cumulative future reward from time $t$, or sum of future rewards).

---

## 4.6 Glossary

* **Stochastic Policy:**  A policy that outputs a probability distribution over actions, allowing for natural exploration.
* **Gradient Ascent:** An optimization algorithm used to find the local maximum of a function (moving in the direction of the gradient).
* **Trajectory ($\tau$):** A full sequence of states, actions, and rewards from the start of an episode to the end.
* **Log-Derivative Trick:** A mathematical identity.trick used to transform the gradient of a probability distribution into an expectation that can be sampled.
* **REINFORCE:** A basic Monte Carlo policy gradient algorithm that uses full episode returns to update policy parameters.

[Understanding Policy Gradient Theorem](https://www.youtube.com/watch?v=cQfOQcpYRzE)

This video provides a visual and intuitive breakdown of how the Policy Gradient Theorem turns complex trajectory probabilities into a simple, implementable formula.
