# Unit 8: Proximal Policy Optimization (PPO)

## Table of Contents
1. [Introduction](#81-introduction)
2. [The Intuition Behind PPO](#82-intuition-behind-ppo)
3. [The Clipped Surrogate Objective](#83-clipped-surrogate-objective)
    * [The Probability Ratio](#the-probability-ratio)
    * [The Clipping Mechanism](#the-clipping-mechanism)
    * [The Final Objective Function](#the-final-objective-function)
4. [Visualize the Clipped Surrogate Objective Function](#84-visualize-the-clipped-surrogate-objective-function)
5. [Glossary](#85-glossary)

---

## 8.1 Introduction
**Proximal Policy Optimization (PPO)** is currently the "gold standard" algorithm at OpenAI and in the RL community. It is an **Actor-Critic** method that addresses a major flaw in standard Policy Gradients: **Stability**.

In algorithms like REINFORCE or A2C, a single bad update (a large step in the wrong direction) can collapse the policy, making it impossible for the agent to recover. PPO ensures that the update is "proximal"; meaning the new policy doesn't stray too far from the old one.

PPO is an architecture that **improves our agent’s training stability by avoiding policy updates that are too large.** To do that, we use a ratio that indicates the difference between our current and old policy and clip this ratio to a specific range $[1 - \epsilon, 1+ \epsilon]$.

---

## 8.2 Intuition Behind PPO
The core idea is **Trust Region Optimization**. We want to improve the policy as much as possible, but we don't want to change it so drastically that we fall off a "numerical cliff."

Standard Policy Gradient calculates the gradient based on the current version of the policy. PPO, however, looks at the **ratio** between the new policy and the old policy. 



* **On-Policy Learning:** PPO is technically on-policy, but it uses "mini-batch" updates on collected experience, making it much more sample-efficient than REINFORCE.

---

## 8.3 The Clipped Surrogate Objective

In REINFORCE, the idea was that by taking a gradient ascent step on this function (equivalent to taking gradient descent of the negative of this function), we would **push our agent to take actions that lead to higher rewards and avoid harmful actions.**

However, the problem comes from the step size:

* Too small, **the training process was too slow**
* Too high, **there was too much variability in the training**

With PPO, the idea is to constrain our policy update with a new objective function called the _Clipped surrogate objective function_ that **will constrain the policy change in a small range using a clip.** This new function $L^{CLIP}(\theta)$ **is designed to avoid destructively large weights updates.**

### The Probability Ratio
The probability ratio is an easy way to estimate the divergence between old and current policy. Instead of just using $\log \pi(a|s)$, PPO defines a ratio $r_t(\theta)$:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

* If $r_t(\theta) > 1$: The action is more likely in the new/current policy than the old.
* If $0 < r_t(\theta) < 1$: The action is more likely for the old policy than the current one.
* If $r_t(\theta) = 1$: The policies are identical.

### The Clipping Mechanism
If we simply maximize $r_t(\theta) \hat{A}_t$, the update could be massive. PPO "clips" this ratio to be within a small interval (usually $[0.8, 1.2]$), defined by a hyperparameter $\epsilon$.

The **Clipped Surrogate Objective** is:

$$L^{CLIP}(\theta) = \hat{E}_t [ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta) 1-\epsilon, 1+\epsilon) \hat{A}_t) ]$$

The unclipped part of the CLipped Surrogate objective called **Conservative Policy Iteration** is:

$$L^{CPI}(\theta) = \hat{E}_t [ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t ] = \hat{E}_t [r_t(\theta) \hat{A}_t]$$

**How it works:**
1. **Positive Advantage ($\hat{A}_t > 0$):** The action was better than average. We want to increase its probability, but we stop increasing it once the ratio $r_t(\theta)$ exceeds $1+\epsilon$.
2. **Negative Advantage ($\hat{A}_t < 0$):** The action was worse than average. We want to decrease its probability, but we stop once the ratio falls below $1-\epsilon$.



### The Final Objective Function
Similar to A2C, PPO combines the policy loss, value loss, and entropy:

$$L_t^{PPO}(\theta) = \hat{E}_t [ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi\_\theta](s_{t}) ]$$

$$L_t^{PPO}(\theta)$$

$$\hat{\mathbb{E}}_t$$

$$L_t^{CLIP}(\theta)$$

$$c_1 L_t^{VF}(\theta)$$

$$c_2 S[\pi_\theta](s_t)$$

$$c_2 S[\pi_\theta](s\_t)$$

$$S[\pi_\theta](s\_t)$$

$$(s\_t)$$

$$S[\pi_\theta]$$

$$c_2$$

* $L_t^{VF}$: Value function (Critic) error (MSE).
* $S$: Entropy bonus to encourage exploration.
* $c_1, c_2$: Hyperparameter coefficients.
* $\hat{\mathbb{E}}_t$: The empirical expectation over a finite batch of samples.
* $L^{CLIP}_t(\theta)$: The clipped surrogate policy objective (moves the Actor).
* $c_1 L^{VF}_t(\theta)$: The value function loss (updates the Critic to minimize error).
* $c_2 S[\pi_\theta](s\_t)$: The entropy bonus (encourages exploration by penalizing a deterministic policy).

---

## 8.4 Visualize the Clipped Surrogate Objective Function
Visualization of PPO training often shows a much smoother "reward curve" compared to DQN or REINFORCE. 

* **The Constraint:** By keeping the update within the $[1-\epsilon, 1+\epsilon]$ "Trust Region," the policy updates are incremental.
* **The "Flat" Region:** When the objective function flattens out due to clipping, the gradient becomes zero for those specific transitions. This prevents the "over-optimization" of a single batch of data.

---

## 8.5 Glossary
* **Trust Region:** The area around the current policy where we believe the mathematical approximation of the gradient is accurate.
* **Surrogate Objective:** An objective function that is not the true objective ($J(\theta)$) but behaves similarly and is easier to optimize.
* **Ratio ($r_t$):** The relationship between the probability of an action under the current policy vs. the previous policy.
* **Clipping:** The process of forcing a value to stay within a specific range to prevent extreme updates.
* **Epsilon ($\epsilon$):** The hyperparameter that defines the size of the trust region (commonly 0.1 or 0.2).
* **Sample Efficiency:** A measure of how much an agent can learn from a limited amount of experience.