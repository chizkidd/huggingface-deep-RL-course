# Unit 8: Proximal Policy Optimization (PPO)

## Table of Contents
1. [Introduction](#1-introduction)
   - 1.1 [Where We Stand](#11-where-we-stand)
   - 1.2 [The Problem with Policy Gradients](#12-the-problem-with-policy-gradients)
   - 1.3 [Enter PPO: Proximal Policy Optimization](#13-enter-ppo-proximal-policy-optimization)
   - 1.4 [What You'll Learn](#14-what-youll-learn)
   - 1.5 [Why PPO Matters](#15-why-ppo-matters)
2. [The Intuition Behind PPO](#2-the-intuition-behind-ppo)
   - 2.1 [The Core Problem: Large Policy Updates](#21-the-core-problem-large-policy-updates)
   - 2.2 [Empirical Evidence for Smaller Updates](#22-empirical-evidence-for-smaller-updates)
   - 2.3 [PPO's Solution: Conservative Policy Updates](#23-ppos-solution-conservative-policy-updates)
   - 2.4 [Comparison: REINFORCE vs A2C vs PPO](#24-comparison-reinforce-vs-a2c-vs-ppo)
3. [The Clipped Surrogate Objective Function](#3-the-clipped-surrogate-objective-function)
   - 3.1 [Recap: Policy Gradient Objective (REINFORCE)](#31-recap-policy-gradient-objective-reinforce)
   - 3.2 [The Ratio Function](#32-the-ratio-function)
   - 3.3 [The Unclipped Surrogate Objective](#33-the-unclipped-surrogate-objective)
   - 3.4 [The Clipped Surrogate Objective](#34-the-clipped-surrogate-objective)
   - 3.5 [Understanding the Minimum Operator](#35-understanding-the-minimum-operator)
   - 3.6 [Complete PPO Objective](#36-complete-ppo-objective)
4. [Visualizing the Clipped Surrogate Objective](#4-visualizing-the-clipped-surrogate-objective)
   - 4.1 [The Six Cases](#41-the-six-cases)
   - 4.2 [Case 1: Ratio in Range, Positive Advantage](#42-case-1-ratio-in-range-positive-advantage)
   - 4.3 [Case 2: Ratio in Range, Negative Advantage](#43-case-2-ratio-in-range-negative-advantage)
   - 4.4 [Case 3: Ratio Below Range, Positive Advantage](#44-case-3-ratio-below-range-positive-advantage)
   - 4.5 [Case 4: Ratio Below Range, Negative Advantage](#45-case-4-ratio-below-range-negative-advantage)
   - 4.6 [Case 5: Ratio Above Range, Positive Advantage](#46-case-5-ratio-above-range-positive-advantage)
   - 4.7 [Case 6: Ratio Above Range, Negative Advantage](#47-case-6-ratio-above-range-negative-advantage)
   - 4.8 [Summary Table](#48-summary-table)
   - 4.9 [Graphical Visualization](#49-graphical-visualization)
   - 4.10 [Why This Design Works](#410-why-this-design-works)
5. [Glossary](#5-glossary)
   - 5.1 [Core PPO Concepts](#51-core-ppo-concepts)
   - 5.2 [Mathematical Components](#52-mathematical-components)
   - 5.3 [Training Components](#53-training-components)
   - 5.4 [Comparison Terms](#54-comparison-terms)
   - 5.5 [Related Algorithms](#55-related-algorithms)
   - 5.6 [Implementation Details](#56-implementation-details)
   - 5.7 [PPO Variants](#57-ppo-variants)
   - 5.8 [Applications](#58-applications)
   - 5.9 [Advantages of PPO](#59-advantages-of-ppo)
   - 5.10 [Limitations of PPO](#510-limitations-of-ppo)
   - 5.11 [Key Equations Summary](#511-key-equations-summary)

---

## 1. Introduction

### 1.1 Where We Stand

In Unit 6, we learned **Advantage Actor-Critic (A2C)**, a hybrid architecture that combines:
- **Actor** (policy-based): Controls agent behavior
- **Critic** (value-based): Evaluates actions taken

A2C helps stabilize training by reducing variance with the advantage function.

### 1.2 The Problem with Policy Gradients

**Policy gradient methods** (REINFORCE, A2C) have a fundamental challenge:

**Step Size Sensitivity**:
- **Too small** → training is painfully slow
- **Too large** → massive variability, training instability, can "fall off the cliff"

**The Cliff Problem**:
```
Policy Performance

     ↑
Good |     ___
     |    /   
     |   /     
Bad  |  /________  ← One bad update causes catastrophic drop
     |_________________→
        Policy updates
```

**What happens**:
- A single large policy update can drastically worsen performance
- Recovery may take a long time or be impossible
- Training becomes unstable and unpredictable

### 1.3 Enter PPO: Proximal Policy Optimization

**Core Idea**: Constrain policy updates to ensure they stay in a "safe" region

**Key Innovation**: Use a **clipped surrogate objective** that:
- Limits how much the policy can change per update
- Prevents destructively large updates
- Maintains training stability

**The Name "Proximal"**:
- "Proximal" = nearby, close to
- New policy stays *proximal* (close) to old policy
- Avoids drastic policy changes

**Clipping Mechanism**:

The policy update is clipped to range $[1-\epsilon, 1+\epsilon]$:
$$\text{ratio} \in [1-\epsilon, 1+\epsilon]$$

Typical value: $\epsilon = 0.2$ (allows 20% change)

### 1.4 What You'll Learn

**Part 1** (This Unit):
- Theory behind PPO
- Clipped surrogate objective function
- Mathematical derivations
- Code PPO from scratch using CleanRL
- Train on LunarLander-v2

**Part 2** (Next Unit):
- Advanced PPO optimization
- Sample-Factory framework
- Train agents in VizDoom (Doom environments)

### 1.5 Why PPO Matters

**Current State-of-the-Art**:
- One of the most popular RL algorithms
- Used in major breakthroughs (OpenAI Five, ChatGPT RLHF)
- Robust, sample-efficient, easy to implement

**Applications**:
- Robotics
- Game AI
- Language model fine-tuning (RLHF)
- Autonomous systems

---

## 2. The Intuition Behind PPO

### 2.1 The Core Problem: Large Policy Updates

**Empirical Observation**: Smaller, incremental policy updates are more likely to converge to optimal solutions.

**Why Large Updates Are Dangerous**:

1. **Non-Convex Optimization Landscape**
   - Policy space is not convex
   - Large steps can jump over good solutions
   - May land in poor regions of policy space

2. **The Cliff Analogy**
   ```
   Quality
     ↑
     |        Peak
     |       /\
     |      /  \
     |     /    \_____ Plateau
     |    /
     |___/______________ Cliff!
         Policy space →
   ```

   - Small steps: gradually climb to peak
   - Large steps: might jump off cliff, hard to recover

3. **Compounding Errors**
   - One bad update affects all future updates
   - Subsequent updates build on corrupted policy
   - Snowball effect of deterioration

### 2.2 Empirical Evidence for Smaller Updates

**Research Findings**:
- Schulman et al. (2017): Smaller trust regions improve convergence
- Large updates often lead to:
  - Performance collapse
  - Training instability
  - Inability to recover

**Ideal Update Size**:
- Not too small (slow progress)
- Not too large (instability)
- PPO's $\epsilon = 0.2$ is empirically effective

### 2.3 PPO's Solution: Conservative Policy Updates

**Mechanism**: Measure and constrain policy change

**Policy Ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

Where:
- $\pi_\theta$ = current policy
- $\pi_{\theta_{\text{old}}}$ = old policy (before update)
- $r_t(\theta)$ = ratio measuring policy change

**Interpretation of Ratio**:

$$r_t(\theta) \begin{cases}
> 1 & \text{Action more likely in new policy} \\
= 1 & \text{No change in action probability} \\
< 1 & \text{Action less likely in new policy}
\end{cases}$$

**Clipping the Ratio**:

$$r_t^{\text{clip}}(\theta) = \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$$

$$= \begin{cases}
1 - \epsilon & \text{if } r_t(\theta) < 1-\epsilon \\
r_t(\theta) & \text{if } 1-\epsilon \leq r_t(\theta) \leq 1+\epsilon \\
1 + \epsilon & \text{if } r_t(\theta) > 1+\epsilon
\end{cases}$$

**Intuition**:
- Ratio close to 1 → policies are similar (good)
- Ratio far from 1 → policies diverging (clip it!)
- Clipping removes incentive to go too far

### 2.4 Comparison: REINFORCE vs A2C vs PPO

| Method | Update Rule | Stability | Sample Efficiency |
|--------|-------------|-----------|------------------|
| **REINFORCE** | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot G_t]$ | Low | Low |
| **A2C** | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot A_t]$ | Medium | Medium |
| **PPO** | Uses clipped objective (see below) | **High** | **High** |

**PPO's Advantage**:
- More stable than REINFORCE and A2C
- Can reuse experience (multiple epochs per batch)
- Prevents catastrophic policy updates
- Easier to tune hyperparameters

---

## 3. The Clipped Surrogate Objective Function

### 3.1 Recap: Policy Gradient Objective (REINFORCE)

**Standard Policy Gradient**:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

Where:
- $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ = return
- We perform gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

**With Advantage** (Actor-Critic):

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) \cdot A_t \right]$$

Where $A_t = Q(s_t, a_t) - V(s_t)$ or $A_t \approx \delta_t$ (TD error)

**Problem**: No constraint on how much policy can change per update

### 3.2 The Ratio Function

**Definition**:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**Expanded Form**:
- For discrete actions: Direct probability ratio
- For continuous actions (Gaussian policy):

$$r_t(\theta) = \frac{\exp\left(-\frac{(a_t - \mu_\theta(s_t))^2}{2\sigma^2}\right)}{\exp\left(-\frac{(a_t - \mu_{\theta_{\text{old}}}(s_t))^2}{2\sigma^2}\right)}$$

**Why Use the Ratio**:

**Original objective**:
$$J(\theta) = \mathbb{E}[\log \pi_\theta(a_t|s_t) \cdot A_t]$$

**Ratio-based objective** (equivalent for small updates):
$$J(\theta) = \mathbb{E}[r_t(\theta) \cdot A_t]$$

**Derivation** (informal):
$$\nabla_\theta \log \pi_\theta(a_t|s_t) = \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)}$$

When we multiply by $\pi_{\theta_{\text{old}}}$ (constant w.r.t. $\theta$):

$$\nabla_\theta \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \cdot A_t \right] \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t$$

(for small changes)

**Benefits of Ratio Form**:
1. Easy to measure divergence between policies
2. Interpretable ($r=1$ means no change)
3. Enables clipping mechanism

### 3.3 The Unclipped Surrogate Objective

**Formula**:

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \cdot A_t \right]$$

Where CPI = Conservative Policy Iteration

**Expanded**:

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \cdot A_t \right]$$

**Interpretation**:
- If $A_t > 0$ (good action): increase $\pi_\theta(a_t|s_t)$ → $r_t(\theta) \uparrow$
- If $A_t < 0$ (bad action): decrease $\pi_\theta(a_t|s_t)$ → $r_t(\theta) \downarrow$

**Problem**: Without constraints, $r_t(\theta)$ can become very large or very small
- If $r_t(\theta) \gg 1$ and $A_t > 0$ → huge policy update (dangerous!)
- If $r_t(\theta) \ll 1$ and $A_t < 0$ → huge policy update (dangerous!)

### 3.4 The Clipped Surrogate Objective

**PPO's Innovation**: Clip the ratio to prevent excessive updates

**Formula**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) \cdot A_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t\right) \right]$$

**Breaking It Down**:

**Two terms**:
1. **Unclipped**: $r_t(\theta) \cdot A_t$
2. **Clipped**: $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t$

**The minimum operator**: Take the more conservative (lower) of the two
- Acts as a pessimistic bound
- Prevents overly optimistic policy updates

**The Clip Function**:

$$\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases}
1 - \epsilon & \text{if } r < 1-\epsilon \\
r & \text{if } 1-\epsilon \leq r \leq 1+\epsilon \\
1 + \epsilon & \text{if } r > 1+\epsilon
\end{cases}$$

**Typical Value**: $\epsilon = 0.2$

**Visualization of Clipping**:

```
clip(r, 0.8, 1.2)
    
    1.2 |     ___________  ← Flat (clipped)
        |    /
     r  |   /
        |  /
    0.8 |_/_______________  ← Flat (clipped)
        0.8   1.0   1.2
              Original r
```

### 3.5 Understanding the Minimum Operator

**Why Take the Minimum**?

**Goal**: Remove incentive for large policy changes

**Scenario 1**: $A_t > 0$ (good action, want to encourage)
- **Unclipped**: $r_t \cdot A_t$ increases as $r_t$ increases
- **Clipped**: $(1+\epsilon) \cdot A_t$ is maximum (flat after $r_t > 1+\epsilon$)
- **Minimum**: Caps at $(1+\epsilon) \cdot A_t$ → no benefit to increasing $r_t$ beyond $1+\epsilon$

**Scenario 2**: $A_t < 0$ (bad action, want to discourage)
- **Unclipped**: $r_t \cdot A_t$ becomes more negative as $r_t$ decreases
- **Clipped**: $(1-\epsilon) \cdot A_t$ is minimum (flat below $r_t < 1-\epsilon$)
- **Minimum**: Caps at $(1-\epsilon) \cdot A_t$ → no benefit to decreasing $r_t$ beyond $1-\epsilon$

**Effect**: Creates a "trust region" where updates are allowed

### 3.6 Complete PPO Objective

**Full PPO Loss** (Actor-Critic style):

$$L^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ L^{\text{CLIP}}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

**Three Components**:

**1. Clipped Surrogate Objective** (policy loss):
$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right) \right]$$

**2. Value Function Loss** (critic loss):
$$L^{VF}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - V^{\text{target}}_t)^2 \right]$$

Where $V^{\text{target}}_t = r_t + \gamma V(s_{t+1})$ or GAE-computed return

**3. Entropy Bonus** (exploration):
$$S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$$

**Coefficients**:
- $c_1$ = value loss coefficient (typically 0.5)
- $c_2$ = entropy coefficient (typically 0.01)

**Gradient Descent**:
$$\theta \leftarrow \theta - \alpha \nabla_\theta L^{\text{PPO}}(\theta)$$

**Note**: We minimize the negative of clipped objective (equivalent to maximizing)

---

## 4. Visualizing the Clipped Surrogate Objective

### 4.1 The Six Cases

We analyze different scenarios based on:
- **Ratio value**: $r_t(\theta)$ compared to $[1-\epsilon, 1+\epsilon]$
- **Advantage sign**: $A_t$ positive or negative

**Notation**:
- $\epsilon = 0.2$ (so range is $[0.8, 1.2]$)
- Unclipped objective: $L^{\text{unclip}} = r_t \cdot A_t$
- Clipped objective: $L^{\text{clip}} = \text{clip}(r_t, 0.8, 1.2) \cdot A_t$
- **Final objective**: $L = \min(L^{\text{unclip}}, L^{\text{clip}})$

### 4.2 Case 1: Ratio in Range, Positive Advantage

**Conditions**:
- $0.8 \leq r_t \leq 1.2$ (ratio in range)
- $A_t > 0$ (action is good)

**Analysis**:
- Clipping doesn't apply: $\text{clip}(r_t, 0.8, 1.2) = r_t$
- Both objectives are the same: $L^{\text{clip}} = L^{\text{unclip}} = r_t \cdot A_t$
- **Gradient**: Normal policy gradient, will increase $\pi_\theta(a_t|s_t)$

**Outcome**: ✅ **Update policy to increase action probability**

**Example**:
- $r_t = 1.1$, $A_t = 2$
- $L = \min(1.1 \times 2, 1.1 \times 2) = 2.2$
- Gradient pushes $\pi_\theta(a_t|s_t)$ up

### 4.3 Case 2: Ratio in Range, Negative Advantage

**Conditions**:
- $0.8 \leq r_t \leq 1.2$
- $A_t < 0$ (action is bad)

**Analysis**:
- Clipping doesn't apply
- $L^{\text{clip}} = L^{\text{unclip}} = r_t \cdot A_t < 0$
- **Gradient**: Will decrease $\pi_\theta(a_t|s_t)$

**Outcome**: ✅ **Update policy to decrease action probability**

**Example**:
- $r_t = 0.9$, $A_t = -3$
- $L = \min(0.9 \times (-3), 0.9 \times (-3)) = -2.7$
- Gradient pushes $\pi_\theta(a_t|s_t)$ down

### 4.4 Case 3: Ratio Below Range, Positive Advantage

**Conditions**:
- $r_t < 0.8$ (action much less likely in new policy)
- $A_t > 0$ (but action is actually good!)

**Analysis**:
- Clipped: $\text{clip}(r_t, 0.8, 1.2) = 0.8$
- $L^{\text{unclip}} = r_t \cdot A_t$ (small positive)
- $L^{\text{clip}} = 0.8 \cdot A_t$ (larger positive)
- **Minimum**: $L = L^{\text{unclip}} = r_t \cdot A_t$

**Gradient**: Positive, will increase $\pi_\theta(a_t|s_t)$

**Outcome**: ✅ **Update policy to increase action probability** (bring it back to range)

**Intuition**: Policy has moved too far away from old policy (action became unlikely), but action is good → correct this by increasing probability

**Example**:
- $r_t = 0.6$, $A_t = 5$
- $L = \min(0.6 \times 5, 0.8 \times 5) = \min(3, 4) = 3$
- Gradient is positive, increases probability

### 4.5 Case 4: Ratio Below Range, Negative Advantage

**Conditions**:
- $r_t < 0.8$ (action much less likely)
- $A_t < 0$ (action is bad, already decreased)

**Analysis**:
- $L^{\text{unclip}} = r_t \cdot A_t$ (negative, magnitude = $|r_t| \cdot |A_t|$)
- $L^{\text{clip}} = 0.8 \cdot A_t$ (negative, smaller magnitude)
- **Minimum**: $L = L^{\text{clip}} = 0.8 \cdot A_t$

**Gradient**: Zero! (flat line at clipped value)

**Outcome**: ❌ **No update** (action already sufficiently discouraged)

**Intuition**: Action is bad AND already has low probability → no need to decrease further

**Example**:
- $r_t = 0.5$, $A_t = -4$
- $L = \min(0.5 \times (-4), 0.8 \times (-4)) = \min(-2, -3.2) = -3.2$
- At clipped boundary → gradient = 0

**Why Gradient is Zero**:

Taking derivative w.r.t. $\theta$:
$$\frac{\partial}{\partial \theta} [0.8 \cdot A_t] = 0$$

Because $0.8$ and $A_t$ don't depend on $\theta$ (computed from old policy)

### 4.6 Case 5: Ratio Above Range, Positive Advantage

**Conditions**:
- $r_t > 1.2$ (action much more likely in new policy)
- $A_t > 0$ (action is good, already encouraged)

**Analysis**:
- $L^{\text{unclip}} = r_t \cdot A_t$ (large positive)
- $L^{\text{clip}} = 1.2 \cdot A_t$ (capped positive)
- **Minimum**: $L = L^{\text{clip}} = 1.2 \cdot A_t$

**Gradient**: Zero! (flat line)

**Outcome**: ❌ **No update** (action already sufficiently encouraged)

**Intuition**: Action is good AND already has much higher probability → don't be greedy

**Example**:
- $r_t = 1.8$, $A_t = 3$
- $L = \min(1.8 \times 3, 1.2 \times 3) = \min(5.4, 3.6) = 3.6$
- At clipped boundary → gradient = 0

### 4.7 Case 6: Ratio Above Range, Negative Advantage

**Conditions**:
- $r_t > 1.2$ (action much more likely)
- $A_t < 0$ (but action is bad!)

**Analysis**:
- $L^{\text{unclip}} = r_t \cdot A_t$ (large negative)
- $L^{\text{clip}} = 1.2 \cdot A_t$ (smaller negative)
- **Minimum**: $L = L^{\text{unclip}} = r_t \cdot A_t$

**Gradient**: Negative, will decrease $\pi_\theta(a_t|s_t)$

**Outcome**: ✅ **Update policy to decrease action probability** (bring it back to range)

**Intuition**: Policy made bad action too likely → decrease probability

**Example**:
- $r_t = 1.5$, $A_t = -2$
- $L = \min(1.5 \times (-2), 1.2 \times (-2)) = \min(-3, -2.4) = -3$
- Gradient is negative, decreases probability

### 4.8 Summary Table

| Case | $r_t$ vs Range | $A_t$ Sign | Minimum Term | Gradient | Action |
|------|----------------|------------|--------------|----------|---------|
| **1** | In $[0.8, 1.2]$ | $+$ | Either (same) | Non-zero | ✅ Increase $\pi$ |
| **2** | In $[0.8, 1.2]$ | $-$ | Either (same) | Non-zero | ✅ Decrease $\pi$ |
| **3** | $< 0.8$ | $+$ | Unclipped | Non-zero | ✅ Increase $\pi$ |
| **4** | $< 0.8$ | $-$ | **Clipped** | **Zero** | ❌ No update |
| **5** | $> 1.2$ | $+$ | **Clipped** | **Zero** | ❌ No update |
| **6** | $> 1.2$ | $-$ | Unclipped | Non-zero | ✅ Decrease $\pi$ |

**Key Insight**: Policy only updates when:
1. Ratio is in range $[0.8, 1.2]$, OR
2. Ratio is outside range but advantage pushes it back toward range

### 4.9 Graphical Visualization

**For Positive Advantage** ($A_t > 0$):

```
Objective
    ↑
    |        ___________  ← Clipped (flat, no gradient)
    |       /
    |      /            
    |     /              ← Unclipped (increasing)
    |    /               
    |___/________________
        0.8   1.0   1.2  → r_t
        
Update: ✅    ✅    ❌
        (3)   (1)   (5)
```

**For Negative Advantage** ($A_t < 0$):

```
Objective
    ↑
    |____________________
    |    \               
    |     \              ← Unclipped (decreasing)
    |      \            
    |       \___________  ← Clipped (flat, no gradient)
    ↓
        0.8   1.0   1.2  → r_t
        
Update: ❌    ✅    ✅
        (4)   (2)   (6)
```

### 4.10 Why This Design Works

**Prevents Destructive Updates**:
- When policy has already moved far ($r_t$ outside range)
- AND move was in the right direction (good action more likely, bad action less likely)
- Gradient becomes zero → no further push in that direction

**Allows Corrective Updates**:
- When policy moved far in the WRONG direction
- Gradient remains active → pulls policy back

**Creates Trust Region**:
- Policy can freely move within $[1-\epsilon, 1+\epsilon]$
- Beyond that, updates are restricted

**Conservative by Design**:
- Takes minimum of clipped and unclipped
- Pessimistic bound ensures safety
- Prevents overly aggressive optimization

---

## 5. Glossary

### 5.1 Core PPO Concepts

**Proximal Policy Optimization (PPO)**
- State-of-the-art policy gradient algorithm
- Constrains policy updates using clipped surrogate objective
- Prevents destructively large policy changes
- Created by Schulman et al. (OpenAI, 2017)
- Used in: OpenAI Five, ChatGPT fine-tuning, robotics

**Clipped Surrogate Objective**
- Core innovation of PPO
- Objective function: $L^{\text{CLIP}}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$
- Removes incentive for large policy updates
- Creates conservative, stable learning

**Trust Region**
- Safe zone where policy can change freely
- Defined by clip range $[1-\epsilon, 1+\epsilon]$
- Outside this region, updates are restricted
- Balances exploration and stability

**Policy Ratio** $r_t(\theta)$
- Measures how much policy has changed
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
- $r = 1$ means no change
- $r > 1$ means action more likely
- $r < 1$ means action less likely

### 5.2 Mathematical Components

**Surrogate Objective**
- Function that approximates true objective
- Easier to optimize than original
- PPO uses ratio-based surrogate instead of log-probability

**Clip Function**
- $\text{clip}(x, a, b) = \max(a, \min(x, b))$
- Restricts value to range $[a, b]$
- In PPO: $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$

**Epsilon** ($\epsilon$)
- Clip range hyperparameter
- Controls how much policy can change
- Typical value: 0.2 (20% change allowed)
- Range: $[1-\epsilon, 1+\epsilon] = [0.8, 1.2]$

**Advantage Function** $A_t$
- Measures how good action is relative to average
- $A_t = Q(s_t, a_t) - V(s_t)$
- Positive: action better than average
- Negative: action worse than average
- In PPO: Usually computed via GAE

### 5.3 Training Components

**GAE (Generalized Advantage Estimation)**
- Method for computing advantage function
- Balances bias and variance
- $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$
- $\lambda$ controls bias-variance trade-off (typically 0.95)

**Value Function Loss** $L^{VF}$
- Critic's loss for estimating state values
- $L^{VF} = (V_\theta(s_t) - V^{\text{target}}_t)^2$
- Part of total PPO loss

**Entropy Bonus** $S[\pi]$
- Encourages exploration
- $S[\pi_\theta](s) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$
- Higher entropy = more random policy = more exploration
- Coefficient $c_2$ typically 0.01

**Total PPO Loss**
- $L^{\text{PPO}} = L^{\text{CLIP}} - c_1 L^{VF} + c_2 S[\pi]$
- Three components: policy, value, entropy
- Minimized via gradient descent

### 5.4 Comparison Terms

**On-Policy vs Off-Policy**
- **On-policy** (PPO, A2C): Learn from data collected by current policy
- **Off-policy** (DQN, DDPG): Learn from data collected by any policy
- PPO is technically on-policy but reuses data (multiple epochs)

**Sample Efficiency**
- How much experience needed to learn good policy
- PPO: Medium (better than REINFORCE, worse than off-policy methods)
- Improvement over vanilla policy gradients due to data reuse

**Reusing Experience**
- PPO can train on same batch multiple times (multiple epochs)
- Enabled by trust region constraint
- Improves sample efficiency

### 5.5 Related Algorithms

**TRPO (Trust Region Policy Optimization)**
- Predecessor to PPO
- Uses KL divergence constraint: $\mathbb{E}[KL[\pi_{\text{old}} || \pi_\theta]] \leq \delta$
- More theoretically grounded but complex to implement
- Requires conjugate gradient optimization

**A2C (Advantage Actor-Critic)**
- Simpler actor-critic method
- No policy update constraints
- Less stable than PPO but simpler

**A3C (Asynchronous Advantage Actor-Critic)**
- Parallel workers with asynchronous updates
- Precursor to A2C/PPO
- Good for CPU parallelism

### 5.6 Implementation Details

**Minibatch Updates**
- PPO processes data in small batches
- Typical batch size: 64-256
- Multiple epochs over same batch (3-10 epochs)

**Learning Rate**
- Step size for gradient descent
- PPO typically: 3e-4
- Often decayed linearly over training

**Clip Range Decay**
- Some implementations decay $\epsilon$ over training
- Start at 0.2, decay to 0.05
- Makes policy updates more conservative over time

**Normalization**
- Advantage normalization: $\hat{A} = \frac{A - \mu_A}{\sigma_A}$
- Observation normalization: Important for stability
- Reward clipping/normalization

### 5.7 PPO Variants

**PPO-Clip** (Main Version)
- Uses clipped surrogate objective
- What we've studied in this unit
- Most popular PPO variant

**PPO-Penalty** (Alternative)
- Uses KL penalty instead of clipping
- $L = r_t(\theta) A_t - \beta \cdot KL[\pi_{\text{old}} || \pi_\theta]$
- Adaptive $\beta$ based on KL divergence

**Multi-Agent PPO (MAPPO)**
- Extension for multi-agent settings
- Parameter sharing across agents
- Used in cooperative multi-agent tasks

### 5.8 Applications

**Robotics**
- Continuous control tasks
- Locomotion (walking, running)
- Manipulation (grasping, stacking)

**Game AI**
- Dota 2 (OpenAI Five)
- Atari games
- Unity ML-Agents environments

**Language Models**
- RLHF (Reinforcement Learning from Human Feedback)
- ChatGPT fine-tuning
- Aligning LLMs with human preferences

### 5.9 Advantages of PPO

✅ **Stable Training**
- Clipping prevents catastrophic updates
- More reliable convergence than vanilla PG

✅ **Sample Efficient** (for on-policy)
- Can reuse data multiple epochs
- Better than REINFORCE/A2C

✅ **Easy to Implement**
- Simpler than TRPO
- Widely available implementations (Stable-Baselines3, CleanRL)

✅ **Few Hyperparameters**
- Mainly need to tune: learning rate, $\epsilon$, GAE $\lambda$
- Robust to hyperparameter choices

✅ **General Purpose**
- Works for discrete and continuous action spaces
- Single-agent and multi-agent
- Various environment types

### 5.10 Limitations of PPO

❌ **On-Policy Requirement**
- Must collect new data after each policy update
- Less sample-efficient than off-policy methods (SAC, TD3)

❌ **Computationally Expensive**
- Multiple forward/backward passes per batch
- Needs many environment samples

❌ **Sensitive to Reward Scaling**
- Advantage function can have large variance
- Normalization important

❌ **Local Optima**
- Can converge to suboptimal policies
- Careful initialization and tuning needed

### 5.11 Key Equations Summary

| Concept | Equation |
|---------|----------|
| **Policy Ratio** | $r_t(\theta) = \frac{\pi_\theta(a_t\|s_t)}{\pi_{\theta_{\text{old}}}(a_t\|s_t)}$ |
| **Clip Function** | $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ |
| **Clipped Objective** | $L^{\text{CLIP}} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$ |
| **Value Loss** | $L^{VF} = (V_\theta(s) - V^{\text{target}})^2$ |
| **Entropy Bonus** | $S[\pi] = -\sum_a \pi(a\|s) \log \pi(a\|s)$ |
| **Total PPO Loss** | $L^{\text{PPO}} = L^{\text{CLIP}} - c_1 L^{VF} + c_2 S[\pi]$ |
| **GAE** | $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$ |

---

## References

### Academic Papers

- **PPO Original Paper**:
  - Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
  - arXiv:1707.06347

- **TRPO** (Predecessor):
  - Schulman et al. (2015), "Trust Region Policy Optimization"
  - arXiv:1502.05477

- **GAE**:
  - Schulman et al. (2015), "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  - arXiv:1506.02438

### Additional Resources

- **Blog Posts**:
  - Jonathan Hui: "RL — Proximal Policy Optimization (PPO) Explained"
  - OpenAI Spinning Up: PPO Documentation

- **Detailed Explanation**:
  - Daniel Bick (2021): "Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization"
  - Master's thesis, University of Groningen

- **Implementations**:
  - Stable-Baselines3: `pip install stable-baselines3`
  - CleanRL: Clean, single-file implementations
  - RLlib (Ray): Production-ready PPO

### Applications

- **OpenAI Five** (Dota 2): PPO at massive scale
- **ChatGPT/GPT-4**: RLHF with PPO for alignment
- **Unity ML-Agents**: PPO as default algorithm
- **Robotics**: Widely used for continuous control

---

*These notes cover Unit 8: Proximal Policy Optimization (PPO) from the Hugging Face Deep RL Course. PPO is the workhorse of modern reinforcement learning: stable, sample-efficient, and easy to implement.*