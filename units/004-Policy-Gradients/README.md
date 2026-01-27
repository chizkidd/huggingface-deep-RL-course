# Unit 4: Policy Gradient Methods

## Table of Contents
1. [Introduction to Policy-Based Methods](#1-introduction-to-policy-based-methods)
2. [What Are Policy-Based Methods?](#2-what-are-policy-based-methods)
3. [Advantages and Disadvantages](#3-advantages-and-disadvantages)
4. [Deep Dive into Policy Gradients](#4-deep-dive-into-policy-gradients)
5. [The Policy Gradient Theorem](#5-the-policy-gradient-theorem)
6. [The REINFORCE Algorithm](#6-the-reinforce-algorithm)
7. [Implementation Details](#7-implementation-details)
8. [Practical Considerations](#8-practical-considerations)
9. [Glossary](#9-glossary)

---

## 1. Introduction to Policy-Based Methods

### From Value-Based to Policy-Based

Up until this point in the course, we've focused on **value-based methods** where we:
1. Learn a value function (Q-values or state values)
2. Derive a policy from this value function (e.g., ε-greedy policy)

![Value to Policy](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit3/link-value-policy.jpg)

**Key insight from value-based methods**: The policy only exists because of the action-value estimates. The policy is simply a function (like greedy-policy) that selects the action with the highest value.

### The New Paradigm: Policy-Based Methods

With **policy-based methods**, we take a fundamentally different approach:
- **Optimize the policy directly** without needing an intermediate value function
- Learn to map states to probability distributions over actions
- Use gradient ascent to improve the policy parameters

### Why This Matters

Policy-based methods open up new possibilities for:
- Continuous action spaces (e.g., robot joint angles)
- Stochastic policies (naturally explore without ε-greedy)
- Better convergence in certain environments
- High-dimensional action spaces

---

## 2. What Are Policy-Based Methods?

### 2.1 The Core Objective

The fundamental goal of Reinforcement Learning remains unchanged:

> **Find the optimal policy $\pi^*$ that maximizes the expected cumulative reward**

This is based on the **reward hypothesis**:
> All goals can be described as the maximization of expected cumulative reward

**Example**: In a soccer game:
- **Maximize**: Number of goals scored (ball crosses opponent's goal line)
- **Minimize**: Number of goals conceded (ball crosses your goal line)

### 2.2 Three Approaches to Finding Optimal Policy

#### A. Value-Based Methods

**Approach**: Learn a value function → Derive policy

```
Value Function → Policy
Q(s,a) → π(s) = argmax_a Q(s,a)
```

**Characteristics**:
- Learn Q-values or V-values
- Policy is **implicit** (generated from value function)
- Use exploration strategies like ε-greedy
- Examples: Q-Learning, DQN, Double DQN

**Objective**: Minimize loss between predicted and target values
$$L = \mathbb{E}[(Q_{\theta}(s,a) - Q^{target}(s,a))^2]$$

#### B. Policy-Based Methods

**Approach**: Learn policy directly → No value function needed

```
State → Policy Network → Action Probabilities
```

**Characteristics**:
- Parameterize the policy with neural network $\pi_{\theta}$
- Output probability distribution over actions
- Use gradient ascent to optimize parameters
- Examples: REINFORCE, Policy Gradient

**Objective**: Maximize expected return
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$$

![Policy Based](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png)

#### C. Actor-Critic Methods (Preview)

**Approach**: Combine both value-based and policy-based methods

```
Actor (Policy) + Critic (Value Function)
```

- **Actor**: Policy network that selects actions
- **Critic**: Value network that evaluates actions
- Examples: A2C, A3C, PPO, SAC (covered in later units)

### 2.3 Stochastic vs Deterministic Policies

#### Deterministic Policy
$$\pi(s) \rightarrow a$$
- Maps state directly to a single action
- Always takes the same action in the same state
- Common in value-based methods (with greedy policy)

#### Stochastic Policy
$$\pi_{\theta}(a|s) \rightarrow P(a|s)$$
- Outputs probability distribution over actions
- Samples actions from this distribution
- Natural exploration without ε-greedy

![Stochastic Policy](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/stochastic_policy.png)

**Example for CartPole**:
- State: Cart position, velocity, pole angle, angular velocity
- Output: [P(left) = 0.7, P(right) = 0.3]
- Action: Sample from this distribution

### 2.4 Policy-Based vs Policy-Gradient

**Important distinction**:

**Policy-Based Methods** (broader category):
- Optimize policy parameters $\theta$ to maximize objective $J(\theta)$
- Can use **any** optimization technique:
  - Hill climbing
  - Simulated annealing
  - Evolution strategies
  - Gradient-based methods

**Policy-Gradient Methods** (subset):
- Optimize policy parameters using **gradient ascent specifically**
- Compute $\nabla_{\theta} J(\theta)$ and update: $\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$
- Most common and effective approach
- This unit focuses on policy-gradient methods

---

## 3. Advantages and Disadvantages

### 3.1 Advantages of Policy-Gradient Methods

#### Advantage 1: Simplicity of Integration

**No need to store action values**

In value-based methods:
```python
# Must maintain Q-table or Q-network
Q_table[state] = {action1: Q1, action2: Q2, ...}
```

In policy-based methods:
```python
# Direct policy network
action_probs = policy_network(state)
```

**Benefits**:
- Simpler architecture
- Less memory requirements
- Direct optimization of what we care about (policy)

#### Advantage 2: Can Learn Stochastic Policies

**Why stochastic policies matter**:

1. **Automatic Exploration**: No need to implement ε-greedy or other exploration strategies
2. **Solves Perceptual Aliasing**: Can handle situations where different states appear identical

**Example: The Intelligent Vacuum Cleaner**

![Hamster Example](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster1.jpg)

**Setup**:
- Vacuum cleaner must suck dust and avoid killing hamsters
- Can only perceive walls (not exact position)

**The Problem** (Perceptual Aliasing):

![Hamster Aliased States](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster2.jpg)

The two red states are **aliased**:
- Both perceive: "upper wall and lower wall"
- Require different actions: one should go left, other should go right

**Deterministic Policy Failure**:
```
If deterministic: Always go LEFT in red states → Get stuck on the left
                 OR always go RIGHT in red states → Get stuck on the right
```

**Stochastic Policy Success**:
```
Stochastic: P(LEFT|red state) = 0.5, P(RIGHT|red state) = 0.5
Result: Eventually finds dust with high probability
```

![Hamster Solution](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/hamster3.jpg)

#### Advantage 3: Effective in High-Dimensional and Continuous Action Spaces

**The Challenge with Value-Based Methods**:

For continuous actions (e.g., steering wheel angle: 0° to 360°):
- Must compute $Q(s, a)$ for infinite possible actions
- Finding $\max_a Q(s, a)$ becomes an optimization problem itself

**Example: Self-Driving Car**

At each timestep, possible actions:
- Steering wheel: 0.0° to 360.0° (continuous)
- Throttle: 0% to 100% (continuous)
- Brake: 0% to 100% (continuous)

With DQN, you'd need to:
1. Compute Q-value for infinite action combinations
2. Solve optimization problem to find best action
3. Computationally infeasible!

**Policy-Gradient Solution**:
```python
# Direct output: mean and std of action distribution
mean_steering, std_steering = policy_network(state)
action = sample_from_normal(mean_steering, std_steering)
```

#### Advantage 4: Better Convergence Properties

**Value-Based Methods**: Aggressive updates
- Take max over Q-values: $Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a')$
- Small change in Q-values can dramatically change policy
- Can cause instability

**Example**:
```
Step t:   Q(s, LEFT) = 0.22, Q(s, RIGHT) = 0.21  → Choose LEFT (99% of time)
Step t+1: Q(s, LEFT) = 0.21, Q(s, RIGHT) = 0.23  → Choose RIGHT (99% of time)
          Tiny change in values → Dramatic policy change
```

**Policy-Gradient Methods**: Smooth updates
- Action probabilities change gradually
- More stable learning
- Smoother convergence

### 3.2 Disadvantages of Policy-Gradient Methods

#### Disadvantage 1: Converges to Local Maximum

**The Problem**: 
- Policy space is non-convex
- Gradient ascent only guarantees local optima
- May get stuck in suboptimal policies

**Example**:

```
  Global Maximum (Reward = 100)
        ↑
        |     
        |  ← Local Maximum (Reward = 60) ← We might get stuck here
        |     
        |
        └─────────→ Policy Parameters θ
```

**Mitigation Strategies**:
- Use multiple random initializations
- Employ momentum-based optimizers (Adam)
- Add entropy bonuses to encourage exploration

#### Disadvantage 2: Slower Training (Sample Inefficiency)

**Why it's slower**:
- Must collect full episodes before updating
- High variance in gradient estimates
- Requires many samples to get good gradient estimates

**Comparison**:
- **Value-based (DQN)**: Can learn from every transition using replay buffer
- **Policy-gradient**: Must collect full episodes, high variance

**Example Training Time**:
```
CartPole-v1:
- DQN: ~50,000 timesteps to solve
- REINFORCE: ~200,000 timesteps to solve
```

**Mitigation Strategies**:
- Use baseline/advantage functions (Actor-Critic)
- Importance sampling (PPO)
- Better variance reduction techniques

#### Disadvantage 3: High Variance

**The Problem**:
- Returns $R(\tau)$ can vary wildly between episodes
- This variance propagates to gradient estimates
- Leads to unstable training

**Why High Variance Occurs**:

Even with same policy, returns can differ:
```
Episode 1: τ₁ = [s₀, a₀, r₀, s₁, a₁, r₁, ...] → R(τ₁) = 100
Episode 2: τ₂ = [s₀, a₀, r₀, s₁, a₁, r₁, ...] → R(τ₂) = 20
Episode 3: τ₃ = [s₀, a₀, r₀, s₁, a₁, r₁, ...] → R(τ₃) = 150

Gradient estimate is very noisy!
```

**Solutions** (covered in Actor-Critic unit):
- Subtract baseline: Use $R(\tau) - b$ instead of $R(\tau)$
- Use advantage function: $A(s,a) = Q(s,a) - V(s)$
- Multiple parallel actors

---

## 4. Deep Dive into Policy Gradients

### 4.1 The Big Picture

**Goal**: Find parameters $\theta$ that maximize expected return

**Approach**:
1. Parameterize policy with neural network: $\pi_{\theta}(a|s)$
2. Define objective function: $J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$
3. Use gradient ascent to maximize: $\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$

![Policy Gradient Big Picture](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/pg_bigpicture.jpg)

**The Intuition**:
> If we win an episode, we want to increase the probability of actions that led to winning. If we lose, we want to decrease those probabilities.

**The Update Rule** (simplified):
```
For each state-action pair in winning episode:
    Increase P(action|state)
    
For each state-action pair in losing episode:
    Decrease P(action|state)
```

### 4.2 The Parameterized Stochastic Policy

**Definition**: 
$$\pi_{\theta}(a|s) = P_{\theta}(a|s)$$

A neural network that outputs a probability distribution over actions given a state.

**Neural Network Architecture**:

```
Input: State s (e.g., [x, ẋ, θ, θ̇] for CartPole)
         ↓
Hidden Layers (with activation functions)
         ↓
Output Layer (softmax for discrete, gaussian for continuous)
         ↓
Output: Probability distribution over actions
```

**For Discrete Actions** (e.g., CartPole: left or right):
```python
# Network outputs logits
logits = neural_network(state)  # [2.3, 1.1]

# Apply softmax to get probabilities
action_probs = softmax(logits)  # [0.77, 0.23]

# Sample action
action = sample_from_categorical(action_probs)  # 0 (left) with 77% probability
```

**For Continuous Actions** (e.g., robot joint angle):
```python
# Network outputs mean and log_std
mean, log_std = neural_network(state)  # mean=0.5, log_std=-1.0
std = exp(log_std)  # std=0.368

# Sample from Gaussian distribution
action = sample_from_normal(mean, std)  # e.g., 0.45
```

### 4.3 The Objective Function

**Definition**:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$$

The expected cumulative reward (expected return) over all possible trajectories.

**Breaking it Down**:

![Objective Function](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/objective.jpg)

**Components**:

1. **Trajectory $\tau$**: A sequence of states and actions
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$$

2. **Return $R(\tau)$**: Sum of rewards in a trajectory
$$R(\tau) = \sum_{t=0}^{T} r_t$$
or with discount factor:
$$R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$$

3. **Probability of trajectory $P(\tau; \theta)$**: How likely is this trajectory under policy $\pi_{\theta}$?

![Probability](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png)

$$P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)$$

Where:
- $\mu(s_0)$: Initial state distribution
- $P(s_{t+1}|s_t, a_t)$: Environment dynamics (transition probability)
- $\pi_{\theta}(a_t|s_t)$: Policy (action probability)

4. **Expected Return**: Weighted average over all trajectories

![Expected Return](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/expected_reward.png)

$$J(\theta) = \sum_{\tau} P(\tau; \theta) R(\tau)$$

**Interpretation**:
- Sum over all possible trajectories $\tau$
- Weight each trajectory's return by its probability
- This gives us the expected return

**Our Goal**:
$$\theta^* = \arg\max_{\theta} J(\theta)$$

Find parameters $\theta$ that maximize expected return.

![Maximize Objective](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/max_objective.png)

### 4.4 Gradient Ascent

Since we want to **maximize** $J(\theta)$, we use **gradient ascent** (not descent!).

**Gradient Ascent Update Rule**:
$$\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$$

Where:
- $\alpha$: Learning rate (step size)
- $\nabla_{\theta} J(\theta)$: Gradient of objective function with respect to $\theta$

**Gradient Ascent vs Gradient Descent**:

| Gradient Descent | Gradient Ascent |
|-----------------|-----------------|
| Minimize loss | Maximize reward |
| $\theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta)$ | $\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$ |
| Move opposite to gradient | Move in direction of gradient |
| Find minimum | Find maximum |

**Intuition**:
- $\nabla_{\theta} J(\theta)$ points in direction of steepest **increase** of $J(\theta)$
- By moving in this direction, we increase expected return
- Repeat until convergence (hopefully to maximum)

### 4.5 The Challenge: Computing the Gradient

**Problem 1: Computational Intractability**

To compute $\nabla_{\theta} J(\theta)$ exactly:

$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \sum_{\tau} P(\tau; \theta) R(\tau)$$

We would need to:
1. Sum over **all possible trajectories** $\tau$
2. In most environments, there are infinite trajectories
3. Computationally impossible!

**Solution**: Use **sample-based estimation**
- Collect a few trajectories by running the policy
- Approximate the gradient using these samples
- This is Monte Carlo estimation

**Problem 2: Differentiating Environment Dynamics**

The probability of a trajectory depends on:
$$P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)$$

To take gradient:
$$\nabla_{\theta} P(\tau; \theta) = \nabla_{\theta} \left[\mu(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)\right]$$

But we can't differentiate $P(s_{t+1}|s_t, a_t)$ (environment dynamics)!
- We don't have access to environment's transition function
- Even if we did, it's often not differentiable

**Solution**: The **Policy Gradient Theorem** (next section!)

---

## 5. The Policy Gradient Theorem

### 5.1 The Goal

Transform the objective function gradient into a form that:
1. We can estimate from samples (trajectories)
2. Doesn't require differentiating environment dynamics
3. Only requires differentiating our policy $\pi_{\theta}$

### 5.2 The Derivation

**Starting Point**:
$$\nabla_{\theta} J(\theta) = \nabla_{\theta} \sum_{\tau} P(\tau; \theta) R(\tau)$$

#### Step 1: Gradient of Sum = Sum of Gradients

$$\nabla_{\theta} J(\theta) = \sum_{\tau} \nabla_{\theta} [P(\tau; \theta) R(\tau)]$$

Since $R(\tau)$ doesn't depend on $\theta$:
$$= \sum_{\tau} \nabla_{\theta} P(\tau; \theta) \cdot R(\tau)$$

#### Step 2: The Likelihood Ratio Trick (REINFORCE Trick)

**Clever manipulation**: Multiply and divide by $P(\tau; \theta)$

$$= \sum_{\tau} \frac{P(\tau; \theta)}{P(\tau; \theta)} \nabla_{\theta} P(\tau; \theta) \cdot R(\tau)$$

Rearrange:
$$= \sum_{\tau} P(\tau; \theta) \frac{\nabla_{\theta} P(\tau; \theta)}{P(\tau; \theta)} \cdot R(\tau)$$

**Key insight**: Use the log derivative trick
$$\nabla_{\theta} \log f(\theta) = \frac{\nabla_{\theta} f(\theta)}{f(\theta)}$$

Therefore:
$$\frac{\nabla_{\theta} P(\tau; \theta)}{P(\tau; \theta)} = \nabla_{\theta} \log P(\tau; \theta)$$

**Substituting**:
$$\nabla_{\theta} J(\theta) = \sum_{\tau} P(\tau; \theta) \nabla_{\theta} \log P(\tau; \theta) \cdot R(\tau)$$

Now this is an **expectation**!
$$= \mathbb{E}_{\tau \sim P(\tau; \theta)} [\nabla_{\theta} \log P(\tau; \theta) \cdot R(\tau)]$$

#### Step 3: Monte Carlo Estimation

Since it's an expectation, we can approximate with samples:
$$\nabla_{\theta} J(\theta) \approx \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} \log P(\tau^{(i)}; \theta) \cdot R(\tau^{(i)})$$

Where $\tau^{(i)}$ are trajectories sampled by running policy $\pi_{\theta}$.

#### Step 4: Simplifying $\nabla_{\theta} \log P(\tau; \theta)$

Recall:
$$P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t|s_t)$$

Take the log:
$$\log P(\tau; \theta) = \log \mu(s_0) + \sum_{t=0}^{T} \log P(s_{t+1}|s_t, a_t) + \sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t)$$

Take the gradient:
$$\nabla_{\theta} \log P(\tau; \theta) = \nabla_{\theta} \log \mu(s_0) + \nabla_{\theta} \sum_{t=0}^{T} \log P(s_{t+1}|s_t, a_t) + \nabla_{\theta} \sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t)$$

**Critical simplification**: First two terms don't depend on $\theta$!
- $\nabla_{\theta} \log \mu(s_0) = 0$ (initial state distribution)
- $\nabla_{\theta} \sum_{t=0}^{T} \log P(s_{t+1}|s_t, a_t) = 0$ (environment dynamics)

Therefore:
$$\nabla_{\theta} \log P(\tau; \theta) = \nabla_{\theta} \sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$$

### 5.3 The Policy Gradient Theorem

**Final Formula**:

![Policy Gradient Theorem](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_theorem.png)

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R(\tau) \right]$$

**Sample-based approximation** (what we actually compute):
$$\nabla_{\theta} J(\theta) \approx \hat{g} = \frac{1}{m} \sum_{i=1}^{m} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t^{(i)}|s_t^{(i)}) \cdot R(\tau^{(i)})$$

Where:
- $m$: Number of episodes collected
- $\tau^{(i)}$: The $i$-th sampled trajectory
- $R(\tau^{(i)})$: Return of the $i$-th trajectory
- $\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$: Log probability gradient

### 5.4 Intuitive Interpretation

**What does this formula mean?**

$$\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R(\tau)$$

1. **$\nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$**: Direction to increase log probability of action $a_t$ at state $s_t$

2. **$R(\tau)$**: The "score" or "weight" for this direction
   - If $R(\tau)$ is **high** (good episode): Push probabilities **up** for actions in this trajectory
   - If $R(\tau)$ is **low** (bad episode): Push probabilities **down** for actions in this trajectory

**Example**:

```
Good Episode (R(τ) = +100):
    State s₀, Action a₀: θ ← θ + α · ∇_θ log π_θ(a₀|s₀) · (+100)
                         ↑ Increases probability of a₀ at s₀
    State s₁, Action a₁: θ ← θ + α · ∇_θ log π_θ(a₁|s₁) · (+100)
                         ↑ Increases probability of a₁ at s₁
    
Bad Episode (R(τ) = -50):
    State s₀, Action a₀: θ ← θ + α · ∇_θ log π_θ(a₀|s₀) · (-50)
                         ↑ Decreases probability of a₀ at s₀
    State s₁, Action a₁: θ ← θ + α · ∇_θ log π_θ(a₁|s₁) · (-50)
                         ↑ Decreases probability of a₁ at s₁
```

### 5.5 Why This Works

**Key Properties**:

1. **No environment dynamics**: We only need to differentiate our policy $\pi_{\theta}$, not $P(s'|s,a)$

2. **Sample-based**: We can estimate the gradient by collecting trajectories

3. **Unbiased estimator**: The sample-based estimate is an unbiased estimate of the true gradient

4. **Model-free**: We don't need to know the environment dynamics

---

## 6. The REINFORCE Algorithm

### 6.1 Algorithm Overview

**REINFORCE** (Monte Carlo Policy Gradient) is the simplest policy gradient algorithm.

**Key Characteristics**:
- Uses returns from **entire episodes** to estimate gradient
- Monte Carlo method (waits until episode ends)
- On-policy (learns from current policy)
- Also called "Vanilla Policy Gradient"

### 6.2 The Algorithm

**Algorithm: REINFORCE**

```
Initialize policy parameters θ randomly
For episode = 1, 2, 3, ... do:
    1. Generate an episode τ = (s₀, a₀, r₀, ..., s_T, a_T, r_T) using π_θ
    
    2. For t = 0 to T:
           Compute return: R_t = Σ_{t'=t}^{T} γ^{t'-t} r_{t'}
    
    3. Compute gradient estimate:
           ĝ = Σ_{t=0}^{T} ∇_θ log π_θ(a_t|s_t) · R_t
    
    4. Update parameters:
           θ ← θ + α · ĝ
```

**With Multiple Episodes**:

```
For iteration = 1, 2, 3, ... do:
    1. Collect m episodes: {τ⁽¹⁾, τ⁽²⁾, ..., τ⁽ᵐ⁾} using π_θ
    
    2. For each episode i:
           For each timestep t:
               Compute return: R_t⁽ⁱ⁾ = Σ_{t'=t}^{T} γ^{t'-t} r_{t'}⁽ⁱ⁾
    
    3. Compute gradient estimate:
           ĝ = (1/m) Σ_{i=1}^{m} Σ_{t=0}^{T} ∇_θ log π_θ(a_t⁽ⁱ⁾|s_t⁽ⁱ⁾) · R_t⁽ⁱ⁾
    
    4. Update parameters:
           θ ← θ + α · ĝ
```

### 6.3 Return Calculation Options

#### Option 1: Total Episode Return (Simplified REINFORCE)

Use the same return for all timesteps in an episode:
$$R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$$

```python
# Calculate once per episode
total_return = sum([gamma**t * r_t for t, r_t in enumerate(rewards)])

# Use for all timesteps
for t in range(T):
    gradient += grad_log_prob[t] * total_return
```

**Pros**: Simpler implementation
**Cons**: Higher variance (doesn't account for causality)

#### Option 2: Reward-to-Go (Better)

Use return from timestep $t$ onwards:
$$R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$$

```python
# Calculate for each timestep
returns = []
R = 0
for r in reversed(rewards):
    R = r + gamma * R
    returns.insert(0, R)

# Use timestep-specific returns
for t in range(T):
    gradient += grad_log_prob[t] * returns[t]
```

**Pros**: Lower variance (accounts for causality)
**Cons**: Slightly more complex

**Why it's better**: Actions at time $t$ can only affect future rewards, not past rewards!

### 6.4 Pseudocode with Details

```python
def reinforce(env, num_episodes, alpha, gamma):
    """
    REINFORCE algorithm
    
    Args:
        env: Gym environment
        num_episodes: Number of episodes to train
        alpha: Learning rate
        gamma: Discount factor
    """
    # Initialize policy network
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = Adam(policy.parameters(), lr=alpha)
    
    for episode in range(num_episodes):
        # 1. Collect episode
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        while not done:
            # Sample action from policy
            action_probs = policy(state)
            action = sample_from_categorical(action_probs)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # 2. Calculate returns (reward-to-go)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        # 3. Calculate gradient and update
        policy_loss = 0
        for t in range(len(states)):
            # Get action probabilities
            action_probs = policy(states[t])
            
            # Calculate log probability
            log_prob = torch.log(action_probs[actions[t]])
            
            # Accumulate weighted log probability
            policy_loss += -log_prob * returns[t]  # Negative for gradient ascent
        
        # 4. Update policy parameters
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Return: {sum(rewards)}")
    
    return policy
```

### 6.5 Gradient Calculation in Practice

**For Discrete Actions** (e.g., CartPole):

```python
# Forward pass
action_logits = policy_network(state)  # [batch_size, num_actions]
action_probs = softmax(action_logits)  # [batch_size, num_actions]

# Sample action
action_dist = Categorical(action_probs)
action = action_dist.sample()

# Calculate log probability
log_prob = action_dist.log_prob(action)  # This is log π_θ(a|s)

# Weighted by return
loss = -log_prob * return_value  # Negative because we want gradient ascent
```

**For Continuous Actions** (e.g., Robot control):

```python
# Forward pass
mean, log_std = policy_network(state)
std = torch.exp(log_std)

# Sample action from Gaussian
action_dist = Normal(mean, std)
action = action_dist.sample()

# Calculate log probability
log_prob = action_dist.log_prob(action)

# Weighted by return
loss = -log_prob * return_value
```

### 6.6 Complete Training Loop

```python
def train_reinforce():
    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    gamma = 0.99
    num_episodes = 1000
    
    for episode in range(num_episodes):
        # Collect episode
        log_probs, rewards = [], []
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            
            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            # Take action in environment
            state, reward, done, _ = env.step(action.item())
            rewards.append(reward)
        
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        # Normalize returns (reduces variance)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Logging
        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f'Episode {episode}\tTotal Reward: {total_reward:.2f}')
```

---

## 7. Implementation Details

### 7.1 Policy Network Architecture

**For Discrete Actions** (e.g., CartPole: 2 actions):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
```

**For Continuous Actions** (e.g., Robot control):

```python
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
        
        return mean, log_std
```

### 7.2 Return Normalization

**Why normalize returns?**
- Returns can have widely different scales across episodes
- Normalization reduces variance in gradient estimates
- Improves training stability

```python
def normalize_returns(returns):
    """
    Normalize returns to have mean 0 and std 1
    """
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns
```

**Effect**:
```
Before normalization: [100, 50, 200, 75]
After normalization:  [0.26, -0.65, 1.56, -0.13]
```

### 7.3 Gradient Clipping

Prevents exploding gradients:

```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
optimizer.step()
```

### 7.4 Entropy Bonus (Encouraging Exploration)

Add entropy bonus to prevent premature convergence:

$$L(\theta) = -\mathbb{E}[\log \pi_{\theta}(a|s) \cdot R] - \beta H(\pi_{\theta}(·|s))$$

Where entropy:
$$H(\pi_{\theta}(·|s)) = -\sum_{a} \pi_{\theta}(a|s) \log \pi_{\theta}(a|s)$$

```python
# During loss calculation
action_probs = policy(state)
dist = Categorical(action_probs)

# Policy loss
policy_loss = -dist.log_prob(action) * return_value

# Entropy bonus
entropy = dist.entropy()
beta = 0.01  # Entropy coefficient

# Total loss
total_loss = policy_loss - beta * entropy
```

**Effect**: Encourages policy to maintain randomness, preventing premature convergence to deterministic policy.

---

## 8. Practical Considerations

### 8.1 Baseline Subtraction

**Problem**: High variance in gradient estimates

**Solution**: Subtract a baseline $b(s)$ from returns:

$$\nabla_{\theta} J(\theta) \approx \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot [R_t - b(s_t)]$$

**Common baselines**:
1. **Average return**: $b = \frac{1}{m}\sum_{i=1}^{m} R(\tau^{(i)})$
2. **Value function**: $b(s) = V_{\phi}(s)$ (leads to Actor-Critic)

```python
# Calculate baseline (average return)
baseline = np.mean([sum(ep_rewards) for ep_rewards in all_episode_rewards])

# Use in gradient calculation
for log_prob, return_val in zip(log_probs, returns):
    advantage = return_val - baseline
    policy_loss.append(-log_prob * advantage)
```

### 8.2 Handling Episode Length

**For environments with variable episode lengths**:

```python
def collect_episodes(policy, env, num_episodes):
    all_episodes = []
    
    for _ in range(num_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        steps = 0
        max_steps = 1000  # Prevent infinite loops
        
        while not done and steps < max_steps:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        all_episodes.append((states, actions, rewards))
    
    return all_episodes
```

### 8.3 Dealing with Sparse Rewards

**Problem**: Most timesteps have zero reward

**Solutions**:

1. **Reward shaping**: Add intermediate rewards
```python
# Original sparse reward
reward = 1 if goal_reached else 0

# Shaped reward
distance_to_goal = compute_distance(state, goal)
reward = -distance_to_goal + (100 if goal_reached else 0)
```

2. **Reward scaling**: Normalize rewards
```python
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
```

### 8.4 Hyperparameter Tuning

**Key hyperparameters**:

| Hyperparameter | Typical Range | Impact |
|----------------|---------------|--------|
| Learning rate $\alpha$ | 1e-4 to 1e-2 | Too high: instability; Too low: slow learning |
| Discount factor $\gamma$ | 0.95 to 0.999 | Higher = more long-term thinking |
| Episodes per update | 1 to 100 | More = more stable but slower |
| Hidden layer size | 64 to 512 | Larger = more capacity but slower |
| Entropy coefficient $\beta$ | 0.001 to 0.1 | Higher = more exploration |

**Recommended starting values**:
```python
alpha = 1e-3        # Learning rate
gamma = 0.99        # Discount factor
episodes_per_update = 10
hidden_dim = 128
beta = 0.01         # Entropy coefficient
```

### 8.5 Common Pitfalls and Solutions

#### Pitfall 1: No Learning / Flat Performance

**Symptoms**: Policy doesn't improve over time

**Possible causes**:
- Learning rate too low
- Returns not properly calculated
- Gradient not flowing (check with `torch.autograd.grad`)

**Solution**:
```python
# Check gradients
for name, param in policy.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean = {param.grad.mean():.6f}")
```

#### Pitfall 2: Performance Collapse

**Symptoms**: Policy learns then suddenly fails

**Possible causes**:
- Learning rate too high
- No entropy bonus (policy becomes too deterministic)
- Gradient explosion

**Solution**:
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

# Add entropy bonus
loss = policy_loss - 0.01 * entropy_loss
```

#### Pitfall 3: High Variance in Training

**Symptoms**: Reward highly unstable across episodes

**Possible causes**:
- No return normalization
- No baseline subtraction
- Collecting too few episodes per update

**Solution**:
```python
# Normalize returns
returns = (returns - returns.mean()) / (returns.std() + 1e-9)

# Use more episodes per update
episodes_per_update = 20  # Instead of 1
```

---

## 9. Glossary

### Core Concepts

**Policy-Based Methods**
- Reinforcement learning methods that directly learn to approximate the optimal policy
- Do not require learning a value function as an intermediate step
- Output probability distributions over actions

**Policy Gradient**
- A subset of policy-based methods
- Uses gradient ascent to maximize expected return
- Directly optimizes policy parameters $\theta$ by computing $\nabla_{\theta} J(\theta)$

**Stochastic Policy $\pi_{\theta}(a|s)$**
- Policy that outputs a probability distribution over actions
- Parameterized by neural network with weights $\theta$
- Samples actions from this distribution during execution

**Deterministic Policy $\pi(s) \rightarrow a$**
- Policy that outputs a single action for each state
- No randomness in action selection
- Common in value-based methods with greedy policies

### Key Terms

**Trajectory $\tau$**
- Sequence of states, actions, and rewards in an episode
- $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)$

**Return $R(\tau)$**
- Sum of (discounted) rewards in a trajectory
- $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$

**Objective Function $J(\theta)$**
- Expected cumulative reward under policy $\pi_{\theta}$
- $J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$
- What we want to maximize

**Gradient Ascent**
- Optimization algorithm that moves parameters in direction of steepest increase
- Update rule: $\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)$
- Opposite of gradient descent (which minimizes)

**REINFORCE (Monte Carlo Policy Gradient)**
- Simplest policy gradient algorithm
- Uses returns from complete episodes to estimate gradient
- Also called "vanilla policy gradient"

**Policy Gradient Theorem**
- Mathematical result that allows us to compute gradient of expected return
- Eliminates need to differentiate environment dynamics
- Final form: $\nabla_{\theta} J(\theta) = \mathbb{E}[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R(\tau)]$

**Log Derivative Trick (Likelihood Ratio Trick)**
- Mathematical identity: $\nabla_{\theta} \log f(\theta) = \frac{\nabla_{\theta} f(\theta)}{f(\theta)}$
- Key step in deriving policy gradient theorem
- Also called REINFORCE trick

**Reward-to-Go**
- Return calculated from timestep $t$ onwards: $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$
- Accounts for causality (actions only affect future, not past)
- Reduces variance compared to using full episode return

**Baseline**
- Reference value subtracted from returns to reduce variance
- Common choices: average return, value function $V(s)$
- Doesn't introduce bias if chosen correctly

**Entropy Bonus**
- Regularization term encouraging policy randomness
- Prevents premature convergence to deterministic policy
- $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

**Perceptual Aliasing**
- When different states appear identical to the agent
- Problematic for deterministic policies
- Naturally handled by stochastic policies

### Algorithm Comparisons

**Value-Based vs Policy-Based**

| Aspect | Value-Based (e.g., DQN) | Policy-Based (e.g., REINFORCE) |
|--------|-------------------------|--------------------------------|
| What it learns | Q-values or V-values | Policy directly |
| Policy type | Usually deterministic (ε-greedy) | Stochastic |
| Action spaces | Discrete (challenging for continuous) | Handles continuous naturally |
| Sample efficiency | More efficient (replay buffer) | Less efficient (on-policy) |
| Convergence | Can be unstable | Smoother convergence |
| Variance | Lower | Higher |

**On-Policy vs Off-Policy**

- **On-Policy** (`REINFORCE`): Learns from data collected by current policy
- **Off-Policy** (`DQN`): Can learn from data collected by any policy

### Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $s, s_t$ | State at time $t$ |
| $a, a_t$ | Action at time $t$ |
| $r, r_t$ | Reward at time $t$ |
| $\pi_{\theta}(a\|s)$ | Stochastic policy (probability of action $a$ in state $s$) |
| $\theta$ | Policy parameters (neural network weights) |
| $\tau$ | Trajectory (episode) |
| $R(\tau)$ | Return of trajectory |
| $J(\theta)$ | Objective function (expected return) |
| $\nabla_{\theta}$ | Gradient with respect to $\theta$ |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $\mathbb{E}[·]$ | Expected value |
| $P(\tau;\theta)$ | Probability of trajectory under policy $\pi_{\theta}$ |

---

## Summary

Policy gradient methods represent a powerful approach to reinforcement learning that directly optimizes the policy without requiring a value function. Key takeaways:

1. **Direct Policy Optimization**: Learn $\pi_{\theta}(a|s)$ directly rather than deriving policy from values

2. **Stochastic Policies**: Naturally handle exploration and continuous action spaces

3. **Policy Gradient Theorem**: Enables gradient-based optimization without differentiating environment dynamics

4. **REINFORCE Algorithm**: Simplest policy gradient method using Monte Carlo returns

5. **Trade-offs**: 
   - ***PROs:*** Simple, effective for continuous actions, smooth convergence
   - ***CONs:*** High variance, sample inefficient, can converge to local optima

6. **Key Formula**: 
   $$\nabla_{\theta} J(\theta) = \mathbb{E}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \cdot R_t\right]$$

7. **Next Steps**: Actor-Critic methods combine policy gradients with value functions to reduce variance and improve sample efficiency (PPO, A2C, SAC)

The foundation established here paves the way for understanding modern policy gradient algorithms like PPO and SAC, which build on these principles to achieve state-of-the-art performance.
