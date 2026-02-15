# Unit 6: Actor-Critic Methods

- [1. Introduction](#1-introduction)
    - [Where We Left Off: Policy-Based Methods](#11-where-we-left-off-policy-based-methods)
    - [The Core Problem: Variance](#12-the-core-problem-variance)
    - [The Solution: Actor-Critic Methods](#13-the-solution-actor-critic-methods)
    - [What We'll Study](#14-what-well-study)

- [2. The Problem of Variance in Reinforce](#2-the-problem-of-variance-in-reinforce)
    - [Reinforce Recap](#21-reinforce-recap)
    - [What Reinforce Tries to Do](#22-what-reinforce-tries-to-do)
    - [Why This Causes High Variance](#23-why-this-causes-high-variance)
    - [The Unbiasedness Advantage (and Its Cost)](#24-the-unbiasedness-advantage-and-its-cost)
    - [The Naive Fix and Its Problem](#25-the-naive-fix-and-its-problem)
    - [Introducing the Baseline Trick](#26-introducing-the-baseline-trick)

- [3. Advantage Actor-Critic (A2C)](#3-advantage-actor-critic-a2c)
    - [The Actor-Critic Intuition](#31-the-actor-critic-intuition)
    - [Two Function Approximations](#32-two-function-approximations)
    - [The Actor-Critic Training Process (Step by Step)](#33-the-actor-critic-training-process-step-by-step)
    - [Why Actor-Critic Reduces Variance](#34-why-actor-critic-reduces-variance)
    - [Adding the Advantage Function (A2C)](#35-adding-the-advantage-function-a2c)
    - [Full A2C Algorithm](#36-full-a2c-algorithm)
    - [Generalised Advantage Estimation (GAE)](#37-generalised-advantage-estimation-gae)
    - [A2C vs A3C](#38-a2c-vs-a3c)
    - [Comparison: Reinforce vs. Actor-Critic vs. A2C](#39-comparison-reinforce-vs-actor-critic-vs-a2c)
    - [Practical Notes for A2C](#310-practical-notes-for-a2c)

- [4. Glossary](#4-glossary)
    - [Core Algorithms](#41-core-algorithms)
    - [Key Concepts](#42-key-concepts)
    - [Value Functions](#43-value-functions)
    - [Variance and Bias](#44-variance-and-bias)
    - [Exploration](#45-exploration)
    - [Gradient Terms](#46-gradient-terms)
    - [Architecture Terms](#47-architecture-terms)
    - [GAE-Specific Terms](#48-gae-specific-terms)

- [Summary of Key Equations](#summary-of-key-equations)

- [References](#references)

---

## 1. Introduction

### 1.1 Where We Left Off: Policy-Based Methods

In Unit 4, we studied **Reinforce**, our first Policy-Gradient algorithm. It belongs to the family of *Policy-Based Methods* — algorithms that **optimize the policy directly** without an intermediate value function.

**Reinforce** operates via **Gradient Ascent**: it estimates the weights of the optimal policy by pushing up the probability of actions that led to high returns and down for actions that led to low returns.

**Reinforce worked well — but it had a critical weakness: high variance.**

### 1.2 The Core Problem: Variance

Because Reinforce uses **Monte-Carlo sampling** (it collects an entire episode before updating), the return estimate can vary dramatically between episodes due to:
- Stochasticity of the **environment** (random transitions)
- Stochasticity of the **policy** (stochastic action selection)

This leads to **slow, unstable training** — you need enormous numbers of samples to get reliable gradient estimates.

### 1.3 The Solution: Actor-Critic Methods

**Actor-Critic** is a **hybrid architecture** combining:

| Component | Type | Role |
|-----------|------|------|
| **Actor** | Policy-Based | Controls how the agent behaves |
| **Critic** | Value-Based | Measures how good the taken action is |

By adding a Critic that provides feedback at every timestep (rather than waiting for the whole episode), we can dramatically **reduce variance** and stabilize training.

### 1.4 What We'll Study

- The **variance problem** in Reinforce and its root cause
- The **Actor-Critic architecture** and training process
- The **Advantage function** and why it further reduces variance
- The full **A2C (Advantage Actor-Critic)** algorithm
- Training a robotic arm 🦾 with Stable-Baselines3 + Panda-Gym

---

## 2. The Problem of Variance in Reinforce

### 2.1 Reinforce Recap

**Reinforce** (also called REINFORCE or Monte-Carlo Policy Gradient) updates the policy by:

1. Collecting a full trajectory $\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_T)$
2. Computing the discounted return for each timestep
3. Using the return to weight the policy gradient

**The Policy Gradient update rule**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]$$

Where:
- $\theta$ = policy parameters
- $J(\theta)$ = expected return (objective to maximize)
- $\pi_\theta(a_t | s_t)$ = probability of taking action $a_t$ in state $s_t$
- $R(\tau)$ = discounted return of the trajectory

**The return** $R(\tau)$ is computed via Monte-Carlo:

$$R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

### 2.2 What Reinforce Tries to Do

The gradient update essentially says:

- If $R(\tau)$ is **high** → **increase** probability of all $(s_t, a_t)$ pairs in that trajectory
- If $R(\tau)$ is **low** → **decrease** probability of all $(s_t, a_t)$ pairs in that trajectory

**Intuition**: Good trajectories reinforce all the actions that produced them; bad trajectories discourage all actions in them.

### 2.3 Why This Causes High Variance

**Root Cause**: The entire trajectory return $R(\tau)$ is used as a single scalar to update every action in the episode.

**The Problem in Detail**:

Even from the **same starting state**, different episodes can produce very different returns:

```
Episode 1: Start → action A → action B → action C → Total Return = +20
Episode 2: Start → action A → action B → action D → Total Return = -5
Episode 3: Start → action A → action E → action F → Total Return = +8
```

Action A appears in all three episodes with returns of +20, -5, and +8. Its gradient update will fluctuate wildly across episodes.

**Formally**, the variance of a Monte-Carlo return estimate:

$$\text{Var}[R(\tau)] = \mathbb{E}[R(\tau)^2] - (\mathbb{E}[R(\tau)])^2$$

This can be very large because long trajectories accumulate many stochastic events, each introducing noise.

**Two sources of stochasticity**:

1. **Environmental stochasticity**: The same action in the same state doesn't always lead to the same next state
   $$P(s_{t+1} | s_t, a_t) \neq \text{deterministic}$$

2. **Policy stochasticity**: A stochastic policy samples different actions each time
   $$a_t \sim \pi_\theta(\cdot | s_t)$$

### 2.4 The Unbiasedness Advantage (and Its Cost)

**Upside**: Monte-Carlo returns are **unbiased** — we use the true actual return, not an estimate.

$$\mathbb{E}[R(\tau)] = G_t \text{ (true expected return)}$$

**Downside**: True but **high variance**. Every episode gives a slightly (or drastically) different signal.

**The bias-variance trade-off** in RL:

| Method | Bias | Variance | Data Needed |
|--------|------|----------|-------------|
| Monte-Carlo (Reinforce) | None | High | Many episodes |
| TD(0) | Some (bootstrapping) | Low | Few steps |
| Actor-Critic | Some | Lower than MC | Fewer episodes |

### 2.5 The Naive Fix and Its Problem

**Obvious solution**: Collect **more trajectories** per update — the law of large numbers says averaged returns will converge to the true expected return.

$$\hat{\nabla}_\theta J(\theta) = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^n | s_t^n) \cdot R(\tau^n)$$

**Why this fails**: Using a large batch size $N$ massively **reduces sample efficiency**. You need to run many full episodes just to make one gradient update — very expensive.

**Better solution**: Reduce variance at its source using a **baseline** or a **value function** → leads us to Actor-Critic.

### 2.6 Introducing the Baseline Trick

One key tool to reduce variance is to subtract a **baseline** $b(s_t)$ from the return. The policy gradient theorem allows this without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R(\tau) - b(s_t)) \right]$$

**Why is this valid?**  
As long as the baseline $b(s_t)$ doesn't depend on the action $a_t$:

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot b(s_t)\right] = 0$$

So subtracting $b(s_t)$ doesn't change the expected gradient — it only reduces its variance.

**What's the ideal baseline?** The **state value function** $V(s_t)$, which is exactly what the Critic learns.

---

## 3. Advantage Actor-Critic (A2C)

### 3.1 The Actor-Critic Intuition

**Analogy**: Imagine you're learning to play chess.
- **You (Actor)**: Make the moves — you're the policy, deciding which action to take.
- **Your Coach (Critic)**: Watches each move and tells you after each one whether it was good or bad — they're the value function, evaluating actions.

**Without a coach (Reinforce)**: You play an entire game, only finding out at the end if you won or lost, and then try to figure out which moves were responsible.

**With a coach (Actor-Critic)**: After every single move, you get feedback — much faster, more stable learning.

### 3.2 Two Function Approximations

Actor-Critic learns **two neural networks simultaneously**:

**1. The Actor — Policy function** $\pi_\theta(s)$:
$$\pi_\theta : \mathcal{S} \rightarrow \mathcal{P}(\mathcal{A})$$

- Parameterized by $\theta$
- Maps states to a probability distribution over actions
- Updated using the Critic's evaluation
- **Goal**: Find the optimal policy $\pi^*$

**2. The Critic — Value function** $\hat{q}_w(s, a)$:
$$\hat{q}_w : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

- Parameterized by $w$
- Maps (state, action) pairs to their estimated Q-value
- Updated using TD learning
- **Goal**: Accurately evaluate actions taken by the Actor

### 3.3 The Actor-Critic Training Process (Step by Step)

At each timestep $t$:

---

**Step 1: Observe state, compute action and Q-value**

Both networks receive the current state $S_t$ from the environment:

$$A_t \sim \pi_\theta(\cdot | S_t) \quad \text{(Actor samples action)}$$
$$\hat{q}_w(S_t, A_t) \quad \text{(Critic evaluates it)}$$

---

**Step 2: Execute action, observe outcome**

The action $A_t$ is performed in the environment:
- New state: $S_{t+1}$
- Reward: $R_{t+1}$

---

**Step 3: Critic computes TD target and TD error**

Using the Bellman equation, the Critic computes the **TD target**:

$$y_t = R_{t+1} + \gamma \hat{q}_w(S_{t+1}, A_{t+1})$$

And the **TD error** (how wrong the Critic's estimate was):

$$\delta_t = y_t - \hat{q}_w(S_t, A_t) = R_{t+1} + \gamma \hat{q}_w(S_{t+1}, A_{t+1}) - \hat{q}_w(S_t, A_t)$$

---

**Step 4: Actor updates its policy using the Q-value**

The Actor performs a gradient ascent step using the Critic's Q-value:

$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(A_t | S_t) \cdot \hat{q}_w(S_t, A_t)$$

Where $\alpha_\theta$ is the Actor's learning rate.

**Interpretation**: Scale the log-probability gradient by how good the Critic says the action was.

---

**Step 5: Critic updates its value function using TD error**

The Critic minimises the squared TD error:

$$\mathcal{L}(w) = \delta_t^2 = \left(R_{t+1} + \gamma \hat{q}_w(S_{t+1}, A_{t+1}) - \hat{q}_w(S_t, A_t)\right)^2$$

$$w \leftarrow w - \alpha_w \nabla_w \mathcal{L}(w)$$

Where $\alpha_w$ is the Critic's learning rate.

---

**Step 6: Transition and repeat**

$$S_t \leftarrow S_{t+1}$$

Both networks continue updating simultaneously every step.

---

**Full Loop Summary**:

```
S_t
 │
 ├──→ Actor: π_θ(·|S_t) ──→ A_t
 │                              │
 └──→ Critic: q̂_w(S_t, A_t) ←─┘
                  │
           Evaluate action
                  │
          Environment step
                  │
           (R_{t+1}, S_{t+1})
                  │
     ┌────────────┴────────────┐
     ↓                         ↓
  Actor update             Critic update
  θ ← θ + α∇log π·q̂_w     w ← w − α∇L(w)
```

### 3.4 Why Actor-Critic Reduces Variance

**Reinforce** uses the full Monte-Carlo return $R(\tau)$ — which is unbiased but high variance.

**Actor-Critic** uses the **TD target** $R_{t+1} + \gamma \hat{q}_w(S_{t+1}, A_{t+1})$ — which introduces slight bias (from bootstrapping) but has **much lower variance**.

**The trade-off**:

$$\underbrace{R(\tau)}_{\text{Unbiased, High Variance}} \quad \xrightarrow{\text{swap for}} \quad \underbrace{R_{t+1} + \gamma \hat{q}_w(S_{t+1}, A_{t+1})}_{\text{Some Bias, Low Variance}}$$

Lower variance → more stable gradient estimates → faster and more reliable training.

### 3.5 Adding the Advantage Function (A2C)

#### 3.5.1 The Motivation

Using $\hat{q}_w(s, a)$ as the Critic is a good start, but we can do even better by using the **Advantage Function** instead. Why?

Because the raw Q-value $\hat{q}_w(s, a)$ tells us the *absolute* value of taking action $a$ in state $s$ — but not whether that action is *relatively better or worse* than what we'd normally do there.

**Example**:
- $\hat{q}(s, a_1) = 100$, $\hat{q}(s, a_2) = 98$
- $V(s) = 99$ (average value of being in state $s$)
- $a_1$ is only slightly above average (+1), $a_2$ slightly below (-1)

Without the baseline, both get strongly reinforced since both Q-values are high. The Advantage function exposes what truly matters — the **relative** quality.

#### 3.5.2 Formal Definition of the Advantage Function

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

Where:
- $Q(s_t, a_t)$ = expected return starting from $s_t$, taking action $a_t$, then following $\pi$
- $V(s_t)$ = expected return starting from $s_t$ and following $\pi$ (the baseline)
- $A(s_t, a_t)$ = **how much better** (or worse) action $a_t$ is compared to the average

**Sign Interpretation**:

$$A(s, a) \begin{cases} > 0 & \Rightarrow \text{action } a \text{ is better than average — push gradient up} \\ = 0 & \Rightarrow \text{action } a \text{ is exactly average — no change} \\ < 0 & \Rightarrow \text{action } a \text{ is worse than average — push gradient down} \end{cases}$$

#### 3.5.3 The A2C Policy Gradient Update

Replacing $Q(s_t, a_t)$ with the Advantage function in the policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t) \right]$$

**Full update rule**:

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t)$$

This is the **A2C Actor update** — the policy gradient weighted by relative action quality, not absolute return.

#### 3.5.4 The Problem with Naively Computing A(s,a)

The Advantage requires **two separate value functions**:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

This means we'd need:
- One network for $Q(s, a)$ — action-value function
- One network for $V(s)$ — state-value function

That's an extra network — more complexity, more memory, more hyperparameters.

**Solution**: Use the **TD error** as a direct estimator of the Advantage.

#### 3.5.5 TD Error as Advantage Estimator

**Derivation**:

Start with the Advantage definition:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

The Bellman equation for $Q$ gives us:

$$Q(s_t, a_t) = \mathbb{E}[R_{t+1} + \gamma V(s_{t+1})]$$

Substituting:

$$A(s_t, a_t) = \mathbb{E}[R_{t+1} + \gamma V(s_{t+1})] - V(s_t)$$

This is exactly the **TD error** $\delta_t$:

$$\boxed{\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t) \approx A(s_t, a_t)}$$

**Key insight**: The TD error is a single-sample estimate of the Advantage function. We only need **one value network** $V_w(s)$ — much simpler!

**Why this works**:

$$\mathbb{E}_\pi[\delta_t | s_t, a_t] = \mathbb{E}_\pi[R_{t+1} + \gamma V(s_{t+1}) - V(s_t) | s_t, a_t] = Q(s_t, a_t) - V(s_t) = A(s_t, a_t)$$

The TD error is an **unbiased estimator of the Advantage** in expectation.

### 3.6 Full A2C Algorithm

#### 3.6.1 Architecture

**A2C Network Structure** (often shared backbone):

```
         State s_t
             │
    ┌─────────────────┐
    │   Shared Layers │  (CNN for images, MLP for vectors)
    │   Feature Net   │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    ↓                 ↓
 Actor Head       Critic Head
 π_θ(a | s)       V_w(s)
 (policy)         (state value)
 Outputs:          Outputs:
 Action probs      Scalar V(s)
```

**Alternative**: Two fully separate networks (Actor and Critic each have their own full architecture).

#### 3.6.2 Complete Training Loop

```
Initialize policy π_θ and value function V_w

For each episode (or n-step rollout):
    
    Collect trajectory segment: {s_t, a_t, r_{t+1}, s_{t+1}}
    
    For each timestep t in segment:
        
        1. Compute TD error (Advantage estimate):
               δ_t = R_{t+1} + γ V_w(s_{t+1}) - V_w(s_t)
        
        2. Update Critic (minimize squared TD error):
               L_critic = δ_t²
               w ← w - α_w ∇_w L_critic
        
        3. Update Actor (policy gradient with advantage):
               L_actor = -log π_θ(a_t | s_t) · δ_t
               θ ← θ - α_θ ∇_θ L_actor
               (gradient ascent = negative loss, gradient descent)
    
    Repeat
```

#### 3.6.3 The Combined Objective

In practice, A2C optimises a **single combined loss**:

$$\mathcal{L}_{total} = \mathcal{L}_{actor} + c_1 \mathcal{L}_{critic} - c_2 \mathcal{H}[\pi_\theta]$$

**Breaking it down:**

**Actor loss** (we want gradient ascent, so negate for gradient descent):
$$\mathcal{L}_{actor} = -\mathbb{E}_t \left[ \log \pi_\theta(a_t | s_t) \cdot \delta_t \right]$$

**Critic loss** (mean squared TD error):
$$\mathcal{L}_{critic} = \mathbb{E}_t \left[ \delta_t^2 \right] = \mathbb{E}_t \left[ \left(R_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)\right)^2 \right]$$

**Entropy bonus** (encourages exploration by penalising deterministic policies):
$$\mathcal{H}[\pi_\theta] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

**Coefficients**:
- $c_1$ = value loss coefficient (typically 0.5)
- $c_2$ = entropy coefficient (typically 0.01)

**Total loss**:
$$\mathcal{L}_{total} = -\mathbb{E}_t[\log \pi_\theta(a_t|s_t) \cdot \delta_t] + 0.5 \cdot \mathbb{E}_t[\delta_t^2] - 0.01 \cdot \mathcal{H}[\pi_\theta]$$

### 3.7 Generalised Advantage Estimation (GAE)

#### 3.7.1 The n-step Return

Rather than using a 1-step TD error, we can look $n$ steps ahead:

**1-step TD** (low variance baseline, more bias):
$$\delta_t^{(1)} = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

**2-step TD**:
$$\delta_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(s_{t+2}) - V(s_t)$$

**n-step TD** (less bias, more variance):
$$\delta_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l R_{t+l+1} + \gamma^n V(s_{t+n}) - V(s_t)$$

**Monte-Carlo** is just $n = \infty$ (no bootstrapping, zero bias, max variance).

#### 3.7.2 GAE Formula

GAE (Schulman et al., 2015) takes an **exponentially weighted sum** of all n-step advantages:

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

Where:
- $\delta_{t+l} = R_{t+l+1} + \gamma V(s_{t+l+1}) - V(s_{t+l})$
- $\lambda \in [0, 1]$ = GAE trade-off parameter

**Equivalent recursive form**:

$$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$

**Special cases**:

$$\lambda = 0 \Rightarrow \hat{A}_t = \delta_t \quad \text{(1-step TD, low variance, biased)}$$

$$\lambda = 1 \Rightarrow \hat{A}_t = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \approx R(\tau) - V(s_t) \quad \text{(MC, high variance, unbiased)}$$

**Interpretation**: $\lambda$ controls how much you trust the Critic vs. actual experience.

| $\lambda$ | Bias | Variance | Reliance on Critic |
|-----------|------|----------|-------------------|
| 0 | Higher | Lower | High |
| 0.95 | Low | Moderate | Moderate |
| 1 | None | High | None |

### 3.8 A2C vs A3C

| Feature | A2C | A3C |
|---------|-----|-----|
| **Full name** | Advantage Actor-Critic | Asynchronous Advantage Actor-Critic |
| **Workers** | Synchronous (wait for all) | Asynchronous (run independently) |
| **Update** | Batched, synchronized | Immediate per worker |
| **Hardware** | Single machine, GPU-friendly | Multi-core CPU |
| **Gradient** | Averaged across workers | Applied immediately |
| **Reproducibility** | Higher | Lower (async noise) |
| **Paper** | OpenAI (Mnih et al. 2016) | DeepMind (Mnih et al. 2016) |

**A2C is generally preferred today** because GPUs make synchronous batching faster than asynchronous CPU workers.

### 3.9 Comparison: Reinforce vs Actor-Critic vs A2C

| Feature | Reinforce | Actor-Critic | A2C |
|---------|-----------|--------------|-----|
| **Update frequency** | End of episode | Every step | Every step |
| **Critic signal** | None | $Q(s,a)$ | $A(s,a) \approx \delta_t$ |
| **Bias** | None | Low | Low |
| **Variance** | High | Medium | Lower |
| **Networks** | 1 (policy) | 2 (policy + Q) | 2 (policy + V) |
| **Training speed** | Slow | Faster | Fastest |
| **Sample efficiency** | Low | Medium | Higher |

### 3.10 Practical Notes for A2C

#### Shared vs Separate Networks

**Shared network** (common in practice):
- Saves computation
- Forces shared representations
- Needs careful loss balancing ($c_1$, $c_2$)

**Separate networks**:
- Independent learning rates
- Less interference between Actor and Critic
- More parameters, more memory

#### Normalising Advantages

To further stabilise training, **standardise** the advantage values within each minibatch:

$$\hat{A}_{norm} = \frac{\hat{A} - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + \epsilon}$$

This prevents large advantage values from causing excessively large gradient steps.

#### Key Hyperparameters

| Hyperparameter | Typical Value | Effect |
|---------------|---------------|--------|
| `learning_rate` | 3e-4 to 7e-4 | Step size for both networks |
| `gamma` (γ) | 0.99 | Discount factor |
| `gae_lambda` (λ) | 0.95 | GAE trade-off |
| `n_steps` | 5–128 | Steps before update |
| `ent_coef` ($c_2$) | 0.0 to 0.01 | Exploration encouragement |
| `vf_coef` ($c_1$) | 0.25 to 0.5 | Critic loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |

#### Stable-Baselines3 Configuration

```python
from stable_baselines3 import A2C

model = A2C(
    policy="MlpPolicy",        # Or "CnnPolicy" for images
    env=env,
    learning_rate=7e-4,
    n_steps=5,                 # Steps before update
    gamma=0.99,                # Discount factor
    gae_lambda=1.0,            # GAE lambda (1.0 = no GAE)
    ent_coef=0.0,              # Entropy bonus coefficient
    vf_coef=0.5,               # Value function loss coefficient
    max_grad_norm=0.5,         # Gradient clipping
    verbose=1
)

model.learn(total_timesteps=1_000_000)
```

---

## 4. Glossary

### 4.1 Core Algorithms

**Reinforce (REINFORCE)**
- First policy-gradient algorithm studied in the course
- Collects full episodes before updating (Monte-Carlo)
- **Unbiased** but **high variance**
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$

**Actor-Critic**
- Hybrid architecture combining policy-based + value-based
- Two networks: Actor (policy) + Critic (Q-value or V)
- Updates every step (TD learning), not at episode end
- Reduces variance by using a learned value baseline

**A2C (Advantage Actor-Critic)**
- Extends Actor-Critic with the **Advantage function**
- Synchronous variant: workers sync before updating
- Critic learns $V(s)$, uses TD error as Advantage estimate
- Most practical and commonly used Actor-Critic variant today

**A3C (Asynchronous Advantage Actor-Critic)**
- Asynchronous version of A2C
- Multiple workers run independently, update shared model
- Good for CPU parallelism; A2C preferred on GPUs

### 4.2 Key Concepts

**Policy Gradient**
- Class of methods that directly optimize the policy via gradient ascent
- Objective: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$
- Gradient: $\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \cdot \Psi_t]$
- $\Psi_t$ can be: $R(\tau)$, $Q(s,a)$, $A(s,a)$, or TD error

**Monte-Carlo Return**
- Compute return by running full episode to completion
- $R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
- **Unbiased** (uses actual rewards, not estimates)
- **High variance** (sensitive to stochastic events throughout episode)

**TD Error (Temporal Difference Error)**
- One-step estimate of how wrong the value prediction was
- $\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$
- Key dual role: used to update Critic AND as Advantage estimate
- **Low variance** but introduces some **bias** (bootstrapping)

**Bootstrapping**
- Updating a value estimate using another value estimate
- Used by TD methods: $V(s_t) \leftarrow R_{t+1} + \gamma V(s_{t+1})$
- Introduces bias (the estimate $V(s_{t+1})$ may be wrong)
- Reduces variance (only one step of real data needed)

### 4.3 Value Functions

**State-Value Function $V^\pi(s)$**
- Expected cumulative return from state $s$ following policy $\pi$
- $V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$
- Used by the Critic in A2C
- Does not depend on which action is taken

**Action-Value Function $Q^\pi(s,a)$**
- Expected cumulative return from state $s$, taking action $a$, then following $\pi$
- $Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$
- Used by Critic in basic Actor-Critic
- More informative than $V$; more expensive to learn

**Advantage Function $A^\pi(s,a)$**
- Relative value of action $a$ compared to average at state $s$
- $A(s,a) = Q(s,a) - V(s)$
- Positive: action is better than average
- Negative: action is worse than average
- Used as the Critic signal in A2C
- Key: reduces variance by centring the signal

### 4.4 Variance and Bias

**Variance (in RL context)**
- How much the gradient estimate fluctuates across different trajectories
- High variance → unstable training, noisy updates
- Caused by stochastic environments and stochastic policies
- Reinforce suffers from high variance due to Monte-Carlo sampling

**Bias (in RL context)**
- Systematic error introduced by using estimates instead of true values
- Bootstrapping introduces bias (estimates used as targets)
- Low bias means the gradient estimate points in the right direction on average
- Reinforce has zero bias; TD methods have some bias

**Bias-Variance Trade-off**
- The fundamental tension in RL learning
- Reducing variance often introduces bias (and vice versa)
- $\lambda$ in GAE explicitly controls this trade-off
- Goal: find the sweet spot for stable, accurate learning

### 4.5 Exploration

**Entropy $\mathcal{H}[\pi]$**
- Measures how random/spread out the policy distribution is
- $\mathcal{H}[\pi_\theta(·|s)] = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$
- High entropy → exploratory policy (actions roughly equally likely)
- Low entropy → deterministic policy (always picks same action)
- A2C adds entropy bonus to encourage exploration and prevent premature convergence

**Entropy Bonus**
- Added to the A2C objective to prevent the policy from collapsing to deterministic
- $-c_2 \cdot \mathcal{H}[\pi_\theta]$ in the loss (negative because we minimise loss but want to maximise entropy)
- Coefficient $c_2$ controls exploration strength

### 4.6 Gradient Terms

**$\nabla_\theta \log \pi_\theta(a_t|s_t)$ — Score Function**
- Gradient of log-probability of the action taken
- Tells us: "in which direction should I change $\theta$ to make action $a_t$ more or less likely?"
- Foundation of all policy gradient methods
- Scaled by Advantage to determine how much to update

**Gradient Ascent**
- Policy gradient methods maximise the objective $J(\theta)$
- Update: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
- Note: most deep learning uses gradient *descent* to minimise a loss — policy gradient uses *ascent* to maximise return
- In code: equivalent to minimising $\mathcal{L} = -J(\theta)$

**Gradient Clipping**
- Caps gradient norms to prevent large destabilising updates
- $g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|}\right)$
- Typically `max_grad_norm = 0.5` in A2C

### 4.7 Architecture Terms

**Actor**
- The policy network $\pi_\theta(a|s)$
- Takes state as input, outputs action probability distribution
- Updated using the Advantage-weighted policy gradient
- Controls agent behaviour

**Critic**
- The value network $V_w(s)$ (or $Q_w(s,a)$)
- Takes state (and optionally action) as input, outputs scalar value
- Updated using TD error / Bellman equation
- Provides feedback to Actor

**Shared Backbone**
- Single neural network whose output layers branch into Actor head and Critic head
- More efficient than two separate networks
- Requires tuning of loss coefficients $c_1$, $c_2$

### 4.8 GAE-Specific Terms

**GAE (Generalised Advantage Estimation)**
- Weighted combination of all n-step advantage estimates
- $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$
- $\lambda$ controls bias-variance trade-off
- Standard in modern Actor-Critic implementations (A2C, PPO)

**n-step Return**
- Use $n$ actual rewards then bootstrap with value function
- $G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(s_{t+n})$
- Interpolates between TD (n=1) and MC (n=∞)

---

## Summary of Key Equations

| Concept | Formula |
|---------|---------|
| **Monte-Carlo Return** | $R(\tau) = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$ |
| **Reinforce Update** | $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot R(\tau)$ |
| **TD Error** | $\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ |
| **Advantage Function** | $A(s_t,a_t) = Q(s_t,a_t) - V(s_t)$ |
| **Advantage ≈ TD Error** | $A(s_t,a_t) \approx R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ |
| **A2C Actor Update** | $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t\|s_t) \cdot \delta_t$ |
| **A2C Critic Loss** | $\mathcal{L}_{critic} = \delta_t^2$ |
| **Entropy** | $\mathcal{H}[\pi] = -\sum_a \pi(a\|s)\log\pi(a\|s)$ |
| **Total A2C Loss** | $\mathcal{L} = -\log\pi_\theta \cdot \delta_t + c_1\delta_t^2 - c_2\mathcal{H}[\pi_\theta]$ |
| **GAE** | $\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$ |

---

## References

- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning" *(A3C paper)*
- Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" *(GAE paper)*
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, Chapter 13
- [Making Sense of the Bias/Variance Trade-off in Deep RL](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)
- Stable-Baselines3 A2C Documentation

---

*These notes cover Unit 6: Actor-Critic Methods from the Hugging Face Deep RL Course. The core insight is simple: let the Actor learn what to do, let the Critic tell it how well it did — together they train faster and more stably than either could alone.*