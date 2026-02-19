# Unit 7: Multi-Agent Reinforcement Learning (MARL)

# Table of Contents — Unit 7: Multi-Agent Reinforcement Learning (MARL)

## [1. Introduction](#1-introduction)
- 1.1 [From Single-Agent to Multi-Agent Learning](#11-from-single-agent-to-multi-agent-learning)
- 1.2 [The Multi-Agent Reality](#12-the-multi-agent-reality)
- 1.3 [What We'll Learn](#13-what-well-learn)
- 1.4 [Course Maintenance Note](#14-course-maintenance-note)

## [2. Introduction to Multi-Agent Reinforcement Learning](#2-introduction-to-multi-agent-reinforcement-learning)
- 2.1 [Single-Agent vs. Multi-Agent Settings](#21-single-agent-vs-multi-agent-settings)
- 2.2 [Examples of Multi-Agent Environments](#22-examples-of-multi-agent-environments)
- 2.3 [Types of Multi-Agent Environments](#23-types-of-multi-agent-environments)
- 2.4 [Key Differences from Single-Agent RL](#24-key-differences-from-single-agent-rl)

## [3. Designing Multi-Agent Systems](#3-designing-multi-agent-systems)
- 3.1 [The Two Approaches](#31-the-two-approaches)
- 3.2 [Decentralized System (Independent Learners)](#32-decentralized-system-independent-learners)
- 3.3 [Centralized Approach](#33-centralized-approach)
- 3.4 [Hybrid Approaches](#34-hybrid-approaches)
- 3.5 [Comparison Summary](#35-comparison-summary)

## [4. Self-Play](#4-self-play)
- 4.1 [The Challenge of Adversarial Training](#41-the-challenge-of-adversarial-training)
- 4.2 [What is Self-Play?](#42-what-is-self-play)
- 4.3 [Self-Play Algorithm](#43-self-play-algorithm)
- 4.4 [Self-Play in ML-Agents](#44-self-play-in-ml-agents)
- 4.5 [Historical Context](#45-historical-context)
- 4.6 [The ELO Rating System](#46-the-elo-rating-system)
- 4.7 [Self-Play Best Practices](#47-self-play-best-practices)

## [5. Glossary](#5-glossary)
- 5.1 [Core MARL Concepts](#51-core-marl-concepts)
- 5.2 [Environment Types](#52-environment-types)
- 5.3 [Training Approaches](#53-training-approaches)
- 5.4 [Game Theory Concepts](#54-game-theory-concepts)
- 5.5 [Self-Play](#55-self-play)
- 5.6 [Evaluation Metrics](#56-evaluation-metrics)
- 5.7 [Challenges in MARL](#57-challenges-in-marl)
- 5.8 [Algorithms and Techniques](#58-algorithms-and-techniques)
- 5.9 [ML-Agents Specific](#59-ml-agents-specific)
- 5.10 [Key Equations Summary](#510-key-equations-summary)

## [References](#references)

---

**Total Sections:** 5 main + References  
**Total Subsections:** 36 detailed entries  
**Key Equations:** 6+ core formulas  
**Comparison Tables:** 3 (Single vs Multi-Agent, Environment Types, Approaches)  
**Worked Examples:** 2 (ELO calculations, Vacuum cleaners)

---

## 1. Introduction

### 1.1 From Single-Agent to Multi-Agent Learning

Throughout this course, we've trained agents in **single-agent systems**:
- Agent operates alone in its environment
- No cooperation or collaboration with other agents
- All previous units (Q-Learning, DQN, Policy Gradients, Actor-Critic) assumed single-agent settings

**Examples of Single-Agent Environments**:
- Lunar Lander
- CartPole
- Atari games (Breakout, Space Invaders)
- Robotic arm manipulation
- SnowballTarget

### 1.2 The Multi-Agent Reality

**Why Multi-Agent RL Matters**:

1. **Human intelligence emerges from interaction**
   - We learn through social interaction with other agents
   - Collaboration and competition drive development
   - Real-world scenarios rarely involve isolated agents

2. **Real-world applications require multi-agent systems**
   - Autonomous vehicles interacting on roads
   - Warehouse robots coordinating deliveries
   - Game AI (team sports, strategy games)
   - Financial markets (multiple trading agents)
   - Drone swarms

3. **Building robust, adaptive agents**
   - Must handle dynamic environments with other learning agents
   - Need to adapt to opponents' strategies
   - Collaborate effectively with teammates

### 1.3 What We'll Learn

In this unit, we'll study:
- Fundamentals of Multi-Agent Reinforcement Learning (MARL)
- Different types of multi-agent environments
- Centralized vs. Decentralized learning approaches
- **Self-play**: training agents against themselves
- **ELO rating system**: evaluating agent performance
- Training a 2v2 soccer team using Unity ML-Agents

**Practical Application**: SoccerTwos environment
- 2v2 soccer game
- Agents must cooperate with teammates
- Compete against opposing team
- Mixed cooperative-competitive scenario

### 1.4 Course Maintenance Note

**Important**: The AI vs. AI leaderboard for Unit 7 is no longer functional, but:
- You can still train agents
- Observe performance
- Learn all MARL concepts
- Complete the training exercises

---

## 2. Introduction to Multi-Agent Reinforcement Learning

### 2.1 Single-Agent vs. Multi-Agent Settings

#### Single-Agent System

**Characteristics**:
- One agent interacts with environment
- Agent only needs to consider environment dynamics
- Markov Decision Process (MDP) framework

**MDP Components**:
$$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

Where:
- $\mathcal{S}$ = State space
- $\mathcal{A}$ = Action space
- $\mathcal{P}$ = Transition probabilities: $P(s'|s,a)$
- $\mathcal{R}$ = Reward function: $R(s,a,s')$
- $\gamma$ = Discount factor

**Agent's Objective**:
$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

#### Multi-Agent System

**Characteristics**:
- Multiple agents share and interact in common environment
- Agents can observe and influence each other
- Must consider other agents' behaviors and strategies
- More complex than MDP: becomes a **Stochastic Game** (Markov Game)

**Stochastic Game Framework**:
$$\langle n, \mathcal{S}, \{\mathcal{A}_i\}_{i=1}^n, \mathcal{P}, \{\mathcal{R}_i\}_{i=1}^n, \gamma \rangle$$

Where:
- $n$ = Number of agents
- $\mathcal{S}$ = Shared state space
- $\mathcal{A}_i$ = Action space for agent $i$
- $\mathcal{P}$ = Transition probabilities: $P(s'|s, a_1, \ldots, a_n)$
- $\mathcal{R}_i$ = Reward function for agent $i$
- $\gamma$ = Discount factor

**Joint Action Space**:
$$\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_n$$

**Each Agent's Objective**:
$$\pi_i^* = \arg\max_{\pi_i} \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t^i\right]$$

Subject to other agents' policies $\pi_{-i} = (\pi_1, \ldots, \pi_{i-1}, \pi_{i+1}, \ldots, \pi_n)$

### 2.2 Examples of Multi-Agent Environments

#### Warehouse Robots

**Scenario**: Multiple robots navigate warehouse to load/unload packages

**Characteristics**:
- Shared space (warehouse floor)
- Common goal (maximize package throughput)
- Need to avoid collisions
- Coordinate movements

**Challenges**:
- Path planning with dynamic obstacles (other robots)
- Resource allocation (which robot handles which package?)
- Communication/coordination protocols

**State Space**:
$$s_t = [\text{positions}, \text{package\_locations}, \text{robot\_destinations}]$$

#### Autonomous Vehicles

**Scenario**: Several self-driving cars on a highway

**Characteristics**:
- Each car has own destination
- Must respect traffic rules
- Predict other vehicles' behaviors
- Safety-critical decisions

**Challenges**:
- Non-stationary environment (other cars learning/changing behavior)
- Partial observability
- Real-time decision making
- Safety constraints

**State for Vehicle $i$**:
$$s_t^i = [\text{position}^i, \text{velocity}^i, \text{positions}^{-i}, \text{velocities}^{-i}]$$

### 2.3 Types of Multi-Agent Environments

Multi-agent systems can be classified by the relationship between agents:

#### 1. Cooperative Environments

**Definition**: Agents work together to maximize **common benefit** (shared reward)

**Reward Structure**:
$$r_t = r_t^1 = r_t^2 = \ldots = r_t^n \quad \text{(same for all agents)}$$

**Joint Objective**:
$$\max_{\pi_1, \ldots, \pi_n} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

**Examples**:
- **Warehouse robots**: Maximize total packages delivered
- **Multi-robot construction**: Build structure cooperatively
- **Search and rescue**: Multiple drones finding victims

**Key Characteristics**:
- Aligned incentives
- Communication usually beneficial
- Team coordination essential
- Credit assignment problem (who contributed to success?)

**Challenges**:
- Scalability with number of agents
- Coordination overhead
- Emergent behaviors may be suboptimal

#### 2. Competitive/Adversarial Environments

**Definition**: Agents aim to maximize **individual benefit** at expense of opponents

**Reward Structure** (zero-sum):
$$\sum_{i=1}^n r_t^i = 0$$

**Example (Two Players)**:
$$r_t^1 = -r_t^2$$

**Individual Objective**:
$$\max_{\pi_i} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t^i\right] \quad \text{subject to } \pi_{-i}$$

**Examples**:
- **Tennis**: One agent wins, other loses
- **Chess/Go**: Pure competition
- **Poker**: Multiple players competing

**Key Characteristics**:
- Conflicting incentives
- Strategic behavior (deception, adaptation)
- Game-theoretic equilibria (Nash equilibrium)

**Nash Equilibrium**:
A policy profile $(\pi_1^*, \ldots, \pi_n^*)$ where no agent can improve by unilaterally changing strategy:

$$V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i, \pi_{-i}^*) \quad \forall i, \forall \pi_i$$

**Challenges**:
- Non-stationary opponents
- Exploitability vs. robustness trade-off
- Cyclic dynamics (rock-paper-scissors scenarios)

#### 3. Mixed Cooperative-Competitive Environments

**Definition**: Combination of cooperation (within teams) and competition (between teams)

**Reward Structure**:
$$r_t^{\text{team A}} = -r_t^{\text{team B}}$$
$$r_t^{i} = r_t^{\text{team}} \quad \text{for all } i \in \text{team}$$

**Example: Soccer (2v2)**:
- **Team A**: Agents 1, 2 (cooperate)
- **Team B**: Agents 3, 4 (cooperate)
- **Teams compete** for higher score

**Objective for Team A**:
$$\max_{\pi_1, \pi_2} \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t^A\right] \quad \text{subject to } \pi_3, \pi_4$$

**Examples**:
- **SoccerTwos**: 2v2 soccer (our unit's environment)
- **MOBA games**: League of Legends, Dota 2
- **Capture the Flag**: Team-based strategy

**Key Characteristics**:
- Intra-team cooperation
- Inter-team competition
- Complex credit assignment (team vs. individual contribution)
- Strategic diversity

**Challenges**:
- Balancing cooperation and individual skill
- Training stability (multiple levels of learning)
- Emergent team strategies

### 2.4 Key Differences from Single-Agent RL

| Aspect | Single-Agent | Multi-Agent |
|--------|--------------|-------------|
| **Environment** | Stationary (fixed dynamics) | Non-stationary (agents change) |
| **Framework** | MDP | Stochastic Game |
| **Objective** | Maximize individual return | Depends on relationship (coop/comp) |
| **Complexity** | $\|\mathcal{S}\| \times \|\mathcal{A}\|$ | $\|\mathcal{S}\| \times \|\mathcal{A}\|^n$ |
| **Convergence** | Generally guaranteed | Not guaranteed |
| **Observability** | Full (usually) | Partial (often) |
| **Communication** | N/A | May be crucial |

**Curse of Dimensionality in MARL**:
- Joint action space grows exponentially with number of agents
- $\|\mathcal{A}_{\text{joint}}\| = \prod_{i=1}^n \|\mathcal{A}_i\|$
- For $n$ agents with $k$ actions each: $k^n$ joint actions

---

## 3. Designing Multi-Agent Systems

When training multiple agents, we face a fundamental design choice: How should agents learn?

### 3.1 The Two Approaches

We have two main paradigms for training multi-agent systems:

1. **Decentralized Learning** (Independent Learners)
2. **Centralized Learning** (Joint Policy)

Each has distinct advantages and challenges.

### 3.2 Decentralized System (Independent Learners)

#### 3.2.1 Core Concept

**Definition**: Each agent is **trained independently** from the others.

**Key Idea**: From the perspective of agent $i$:
- Other agents are part of the environment
- Agent $i$ doesn't model other agents explicitly
- No information shared between agents during training

**Training Procedure**:
```
For each agent i:
    Initialize policy π_i and value function V_i
    
    For each episode:
        Observe state s_i
        Select action a_i ~ π_i(·|s_i)
        Execute a_i, observe reward r_i and next state s_i'
        Update π_i using standard RL algorithm (Q-Learning, PPO, etc.)
        
    No communication or coordination with other agents during updates
```

#### 3.2.2 Mathematical Formulation

Each agent treats others as part of environment dynamics:

**Agent $i$'s MDP**:
$$\langle \mathcal{S}, \mathcal{A}_i, P_i, R_i, \gamma \rangle$$

Where the transition function $P_i$ implicitly includes other agents' behaviors:

$$P_i(s'|s, a_i) = \sum_{a_{-i}} \left[\prod_{j \neq i} \pi_j(a_j|s)\right] P(s'|s, a_i, a_{-i})$$

**Agent $i$'s Objective**:
$$\max_{\pi_i} \mathbb{E}_{\pi_i, \pi_{-i}}\left[\sum_{t=0}^{\infty} \gamma^t r_t^i\right]$$

#### 3.2.3 Example: Vacuum Cleaners

**Scenario**: 3 vacuum robots in a house

**Decentralized Approach**:
- Each vacuum has its own policy $\pi_i$
- Vacuum 1: "Clean as much as I can, ignoring others"
- Vacuum 2: "Clean as much as I can, ignoring others"
- Vacuum 3: "Clean as much as I can, ignoring others"

**Observation for Vacuum $i$**:
$$o_i = [\text{my\_position}, \text{uncleaned\_areas}, \text{battery\_level}]$$

Note: Doesn't explicitly include other vacuums' positions/states

**Action**: Move to nearest uncleaned area

#### 3.2.4 Advantages

✅ **Simplicity**:
- Each agent uses standard single-agent RL algorithms
- No need for special multi-agent techniques
- Easy to implement and debug

✅ **Scalability**:
- Agents can be added/removed without retraining others
- Each agent trains independently (parallelizable)
- No centralized bottleneck

✅ **Privacy**:
- Agents don't need to share observations/rewards
- Useful when agents belong to different entities

✅ **Robustness**:
- Failure of one agent doesn't affect others' policies
- Can handle heterogeneous agents (different capabilities)

#### 3.2.5 Disadvantages

❌ **Non-Stationary Environment**:

**The Core Problem**: From agent $i$'s perspective, the environment is non-stationary because other agents are also learning and changing their policies.

**Why This Matters**:

At time $t$:
$$P_t(s'|s, a_i) \neq P_{t+1}(s'|s, a_i)$$

Because $\pi_{-i}^{(t)} \neq \pi_{-i}^{(t+1)}$ (other agents updated their policies)

**Impact on Learning**:
- Violates MDP stationarity assumption
- Many RL algorithms assume stationary $P(s'|s,a)$
- Can lead to instability, oscillations, or failure to converge

**Example**:
```
Episode 100: Vacuum 2 goes left when sees Vacuum 1
            → Vacuum 1 learns "going right is good"

Episode 200: Vacuum 2 (now trained) goes right when sees Vacuum 1
            → Vacuum 1's learned policy is now suboptimal
            → Vacuum 1 must relearn
```

❌ **No Coordination**:
- Agents may develop conflicting strategies
- Inefficient overall behavior (e.g., two vacuums cleaning same area)
- Difficulty in tasks requiring tight coordination

❌ **Credit Assignment**:
- Hard to determine if success/failure due to own actions or others' actions
- Noisy learning signals

❌ **Convergence Issues**:
- No guarantee of reaching global optimum
- May converge to poor local equilibrium
- Possible cyclic behaviors (agents constantly adapting to each other)

### 3.3 Centralized Approach

#### 3.3.1 Core Concept

**Definition**: A **single policy is learned** from experiences of **all agents collectively**.

**Key Idea**:
- High-level process collects all agents' experiences
- Learn one policy that controls all agents
- All agents treated as one larger entity

**Training Procedure**:
```
Initialize shared policy π and value function V

For each episode:
    For each timestep:
        For all agents i:
            Observe individual state s_i
        
        Concatenate into joint state: s = [s_1, s_2, ..., s_n]
        
        Sample joint action: a = [a_1, a_2, ..., a_n] ~ π(·|s)
        
        Execute all actions, observe joint reward r (or sum of r_i)
        
        Update π using joint experience (s, a, r, s')
```

#### 3.3.2 Mathematical Formulation

**Joint State**:
$$s_t = [s_t^1, s_t^2, \ldots, s_t^n]$$

**Joint Action**:
$$a_t = [a_t^1, a_t^2, \ldots, a_t^n] \sim \pi(a_t | s_t)$$

**Joint Policy**:
$$\pi: \mathcal{S}^n \rightarrow \Delta(\mathcal{A}_1 \times \ldots \times \mathcal{A}_n)$$

Where $\Delta(\cdot)$ is the probability simplex

**Factored Form** (if independent actions):
$$\pi(a | s) = \prod_{i=1}^n \pi_i(a_i | s)$$

**Centralized Reward** (cooperative):
$$r_t = \sum_{i=1}^n r_t^i \quad \text{or} \quad r_t^{\text{team}}$$

**Objective**:
$$\max_\pi \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

#### 3.3.3 Example: Vacuum Cleaners

**Centralized Approach**:

**Joint Observation**:
$$s = [\text{position}_1, \text{position}_2, \text{position}_3, \text{coverage\_map}, \text{battery\_levels}]$$

**Joint Policy**:
$$\pi(a_1, a_2, a_3 | s) \rightarrow \text{coordinated movement commands}$$

**Central Learner**:
- Observes all vacuums' states
- Decides movements for all three simultaneously
- Optimizes collective cleaning efficiency

**Example Decision**:
- Vacuum 1 → North wing
- Vacuum 2 → South wing
- Vacuum 3 → Recharge station
(No overlap, efficient division of labor)

#### 3.3.4 Advantages

✅ **Stationarity**:
- All agents treated as parts of one system
- Single policy for entire system
- Environment is stationary from the joint policy's perspective

✅ **Guaranteed Convergence** (under standard assumptions):
- Standard RL convergence guarantees apply
- Can prove convergence to optimal joint policy

✅ **Coordination**:
- Natural emergence of coordinated behaviors
- Can explicitly optimize for coordination
- No conflicting strategies

✅ **Shared Information**:
- All agents benefit from each other's experiences
- Sample-efficient for team tasks
- Can leverage global state information

#### 3.3.5 Disadvantages

❌ **Scalability**:

**Exponential Complexity**:
- Joint action space: $|\mathcal{A}| = \prod_{i=1}^n |\mathcal{A}_i|$
- For $n=10$ agents with 5 actions each: $5^{10} \approx 10$ million joint actions
- Intractable for large $n$

**Computational Cost**:
- Centralized computation bottleneck
- Requires high communication bandwidth
- Single point of failure

❌ **Partial Observability**:
- Requires each agent to observe full joint state
- May not be realistic (communication limits, privacy)
- Difficult to execute decentrally

❌ **Homogeneity Assumption**:
- Typically assumes agents have similar capabilities
- Hard to incorporate heterogeneous agents
- May not learn specialized roles naturally

❌ **Training Time**:
- Larger networks (more parameters)
- More complex credit assignment
- Slower convergence

### 3.4 Hybrid Approaches

In practice, many systems use hybrid architectures to get best of both worlds:

#### Centralized Training, Decentralized Execution (CTDE)

**Key Idea**:
- **Training**: Use centralized approach (access to global info)
- **Execution**: Each agent acts based on local observations only

**Algorithms**:
- **QMIX**: Factored value function with mixing network
- **MADDPG**: Multi-Agent DDPG with centralized critic
- **COMA**: Counterfactual Multi-Agent Policy Gradient

**QMIX Formula**:

Factorize joint $Q_{\text{tot}}$ into individual $Q_i$:

$$Q_{\text{tot}}(s, a_1, \ldots, a_n) = g(Q_1(o_1, a_1), \ldots, Q_n(o_n, a_n))$$

Where $g$ is a monotonic mixing function (ensures consistency)

**Benefit**: Scalable execution, coordinated training

### 3.5 Comparison Summary

| Feature | Decentralized | Centralized | CTDE |
|---------|---------------|-------------|------|
| **Stationarity** | ❌ Non-stationary | ✅ Stationary | ✅ Stationary (training) |
| **Scalability** | ✅ High | ❌ Low | ✅ High (execution) |
| **Coordination** | ❌ Poor | ✅ Excellent | ✅ Good |
| **Convergence** | ❌ Not guaranteed | ✅ Guaranteed | ✅ Usually good |
| **Communication** | ✅ Not required | ❌ High | Moderate (training only) |
| **Implementation** | ✅ Simple | ❌ Complex | Moderate |

---

## 4. Self-Play

### 4.1 The Challenge of Adversarial Training

When training agents in **competitive/adversarial** environments (games, sports), we face a chicken-and-egg problem:

**Problem 1: Finding an Opponent**
- Need a competent opponent to train against
- But where do we get this opponent initially?

**Problem 2: Opponent Strength Mismatch**
- If opponent is too weak → agent overfits to beating weak strategies
- If opponent is too strong → agent never wins, can't learn effectively

**Analogy**: Child learning soccer
- Playing against professionals: Too hard, can't learn (always loses)
- Playing against toddlers: Too easy, learns bad habits
- **Ideal**: Play against opponents of similar skill level

### 4.2 What is Self-Play?

**Definition**: Training paradigm where an agent plays against **past versions of itself** as opponents.

**Core Mechanism**:
1. Agent starts with random policy $\pi_0$
2. Plays games against copy of $\pi_0$
3. Updates policy to $\pi_1$ based on experience
4. Now plays against $\pi_1$ (or mixture of $\pi_0, \pi_1$)
5. Updates to $\pi_2$, plays against past policies
6. Repeat: $\pi_0 \rightarrow \pi_1 \rightarrow \pi_2 \rightarrow \ldots \rightarrow \pi^*$

**Key Insight**: The opponent **automatically scales in difficulty** with the learning agent!

### 4.3 Self-Play Algorithm

#### Basic Self-Play Procedure

```
Initialize policy π_0
Initialize opponent pool P = {π_0}

For iteration t = 1, 2, 3, ...:
    
    # Sample opponent
    π_opponent = sample_from_pool(P)
    
    # Play games
    For episode = 1 to N:
        Reset environment
        While not done:
            a_agent = π_t(s)
            a_opponent = π_opponent(s)
            Execute actions, observe rewards
        
        Collect experience: (s, a_agent, r, s')
    
    # Update policy
    π_{t+1} = update(π_t, experience)
    
    # Add to opponent pool (every K iterations)
    If t mod K == 0:
        P = P ∪ {π_t}
```

#### Advanced Self-Play Variants

**1. Prioritized Fictitious Self-Play (PFSP)**:
- Weight opponent sampling by win rate
- Play more against challenging opponents
- $P(\text{select } \pi_i) \propto \text{difficulty}(\pi_i)$

**2. League Training** (AlphaStar):
- Main agents
- Main exploiters (find weaknesses)
- League exploiters (find weaknesses across league)

**3. Population-Based Training**:
- Maintain diverse population
- Evaluate fitness
- Evolve hyperparameters and architectures

### 4.4 Self-Play in ML-Agents

Unity ML-Agents has built-in self-play support with key hyperparameters:

#### 4.4.1 Key Hyperparameters

**1. `save_steps`**: How often to save new opponent

$$\text{save\_steps} = 50000 \text{ steps}$$

**Effect**:
- **Larger value** → wider range of skill levels in pool (from early bad to later good)
- **Smaller value** → more frequent updates, more opponents, but similar skill levels

**2. `swap_steps`**: How often to change current opponent

$$\text{swap\_steps} = 25000 \text{ steps}$$

**Effect**:
- **Larger value** → more stable training (same opponent longer)
- **Smaller value** → more diversity in opponents faced

**3. `window`**: Size of opponent pool

$$\text{window} = 10$$

**Effect**: Only keep last 10 saved policies as potential opponents

**Larger window**:
- Greater diversity of opponent strategies
- Includes policies from earlier training (weaker)
- More robust final policy

**Smaller window**:
- Only recent (stronger) opponents
- Faster improvement but may overfit

**4. `play_against_latest_model_ratio`**: Probability of playing against most recent self

$$P(\text{play against latest}) = 0.5$$

**Effect**:
- **= 1.0** → Always play against latest version (most challenging)
- **= 0.0** → Never play against latest (only older versions)
- **= 0.5** → Balanced (typical)

Higher ratio → faster improvement but less robustness

**5. `team_change`**: Steps before swapping team sides (if applicable)

$$\text{team\_change} = 200000$$

Ensures both teams get trained equally if there's asymmetry

#### 4.4.2 Trade-offs in Self-Play Hyperparameters

**Stability vs. Diversity**:

```
More stable (slower opponent change):
    + Convergence more reliable
    - May overfit to current opponent

More diversity (faster opponent change):
    + More robust policy
    - Training may be unstable
```

**Optimal Configuration** (typical):
```yaml
self_play:
  save_steps: 50000
  swap_steps: 25000
  window: 10
  play_against_latest_model_ratio: 0.5
  team_change: 200000
```

### 4.5 Historical Context

Self-play is not new:

**1950s**: Arthur Samuel's checkers program
- Played against previous versions of itself
- Improved through self-play
- Eventually beat Samuel himself

**1995**: TD-Gammon (Gerald Tesauro)
- Backgammon agent
- Self-play with TD learning
- Reached expert human level

**2016**: AlphaGo (DeepMind)
- Used self-play to master Go
- Defeated world champion Lee Sedol
- Policy network trained via supervised learning + self-play reinforcement

**2017**: AlphaZero (DeepMind)
- Pure self-play (no human games)
- Mastered Chess, Shogi, and Go
- Superhuman in all three from scratch

**2019**: AlphaStar (DeepMind)
- StarCraft II
- League-based self-play
- Grandmaster level

### 4.6 The ELO Rating System

In adversarial settings, **cumulative reward is a poor metric**:
- Depends entirely on opponent strength
- Beating weak opponent → high reward (but meaningless)
- Losing to strong opponent → low reward (but may have played well)

**Solution**: Use relative skill rating system → **ELO**

#### 4.6.1 What is ELO?

**ELO** (named after Arpad Elo): System for calculating relative skill levels in zero-sum games

**Zero-Sum Game**: One agent's gain = other agent's loss
$$r_1 + r_2 = 0$$

**Key Properties**:
1. **Relative**: Your rating depends on opponents' ratings
2. **Dynamic**: Updates after each game
3. **Predictive**: Rating difference predicts win probability

#### 4.6.2 ELO Mathematics

**Expected Score**:

Given two players with ratings $R_A$ and $R_B$:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

$$E_B = \frac{1}{1 + 10^{(R_A - R_B)/400}} = 1 - E_A$$

**Interpretation**:
- $E_A$ = Expected probability that player A wins
- $E_B$ = Expected probability that player B wins

**Properties**:
- $E_A + E_B = 1$ (one must win in zero-sum game)
- Equal ratings: $E_A = E_B = 0.5$
- 400 rating difference ≈ 10:1 win ratio

**Actual Score**:
$$S_A = \begin{cases} 1 & \text{if A wins} \\ 0.5 & \text{if draw} \\ 0 & \text{if A loses} \end{cases}$$

**Rating Update**:

After game, update ratings using:

$$R_A^{\text{new}} = R_A^{\text{old}} + K \cdot (S_A - E_A)$$

Where:
- $K$ = Maximum rating change per game (K-factor)
  - $K=32$ for beginners
  - $K=16$ for masters
- $S_A - E_A$ = performance error

**Intuition**:
- If $S_A > E_A$ (performed better than expected) → rating increases
- If $S_A < E_A$ (performed worse than expected) → rating decreases
- Magnitude of change ∝ surprise (how unexpected the result was)

#### 4.6.3 Worked Example

**Setup**:
- Player A: $R_A = 2600$
- Player B: $R_B = 2300$
- K-factor: $K = 16$

**Step 1: Calculate Expected Scores**

$$E_A = \frac{1}{1 + 10^{(2300-2600)/400}} = \frac{1}{1 + 10^{-0.75}} = \frac{1}{1 + 0.178} = 0.849$$

$$E_B = 1 - E_A = 0.151$$

**Interpretation**: A is expected to win ~85% of the time

**Step 2: Scenario 1 — A Wins (As Expected)**

$$S_A = 1, \quad S_B = 0$$

$$R_A^{\text{new}} = 2600 + 16 \times (1 - 0.849) = 2600 + 2.4 = 2602$$

$$R_B^{\text{new}} = 2300 + 16 \times (0 - 0.151) = 2300 - 2.4 = 2298$$

**Result**: A gains only 2 points (expected to win), B loses 2 points

**Step 3: Scenario 2 — B Wins (Upset!)**

$$S_A = 0, \quad S_B = 1$$

$$R_A^{\text{new}} = 2600 + 16 \times (0 - 0.849) = 2600 - 13.6 = 2586$$

$$R_B^{\text{new}} = 2300 + 16 \times (1 - 0.151) = 2300 + 13.6 = 2314$$

**Result**: A loses 14 points (unexpected loss), B gains 14 points (upset victory)

**Key Insight**: Points transferred = $K \times |S - E|$ are always balanced

#### 4.6.4 ELO in Multi-Agent Training

**During Training**:
```
Initialize: All agents start at ELO = 1200

After each game:
    Calculate E_A, E_B using current ratings
    Observe outcome (S_A, S_B)
    Update R_A, R_B
    Log ELO scores
```

**Expected ELO Progression**:
```
Early training (random policy):
    ELO ≈ 1200 (starting point)
    High variance (random outcomes)

Mid training (learning):
    ELO increasing
    Win rate improving

Late training (converged):
    ELO plateaus at high value
    Consistent wins against older versions
```

**Team ELO** (for team games like soccer):

Average team members' ratings:

$$R_{\text{team}} = \frac{1}{n} \sum_{i \in \text{team}} R_i$$

Use team rating in ELO calculations

#### 4.6.5 Advantages of ELO

✅ **Relative Measure**:
- Ratings have meaning relative to each other
- 100-point difference ≈ 64% win rate for higher-rated player

✅ **Self-Correcting**:
- Winning against weak opponent → small gain
- Losing to strong opponent → small loss
- System balances over time

✅ **Works for Teams**:
- Calculate team average
- Use in same formulas

✅ **Balanced Points**:
- Total points in system remain constant
- Zero-sum: $\Delta R_A + \Delta R_B = 0$

✅ **Interpretable**:
- Can compare agents across training runs
- Track improvement over time

#### 4.6.6 Disadvantages of ELO

❌ **Individual Contribution**:
- In team games, doesn't capture individual performance
- All team members get same rating change

❌ **Rating Inflation/Deflation**:
- Over time, maintaining rating requires improving skill
- New players enter at default rating (affects distribution)

❌ **Transitivity Assumption**:
- Assumes: If A > B and B > C, then A > C
- Not always true in games (rock-paper-scissors scenarios)

❌ **Historical Comparison**:
- Can't directly compare ratings across different time periods
- Rating of 2000 in 1980 ≠ 2000 in 2020

❌ **Initialization Sensitivity**:
- Starting ELO choice affects early dynamics
- Converges eventually, but initial period noisy

### 4.7 Self-Play Best Practices

**Do's**:
1. ✅ Start with balanced `play_against_latest_model_ratio` (0.5)
2. ✅ Use diverse opponent pool (`window` = 5-10)
3. ✅ Save opponents at reasonable intervals (balance diversity and memory)
4. ✅ Monitor ELO to track progress
5. ✅ Use population-based approaches for complex games

**Don'ts**:
1. ❌ Always play against latest (overfitting risk)
2. ❌ Always play against oldest (slow improvement)
3. ❌ Tiny opponent pool (low diversity)
4. ❌ Ignore ELO (blind to actual progress)
5. ❌ Expect monotonic improvement (expect fluctuations)

---

## 5. Glossary

### 5.1 Core MARL Concepts

**Multi-Agent Reinforcement Learning (MARL)**
- Field studying how multiple agents learn in shared environments
- Agents can cooperate, compete, or both
- More complex than single-agent RL due to non-stationarity and coordination

**Stochastic Game** (Markov Game)
- Generalization of MDP to multiple agents
- Tuple: $\langle n, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{\mathcal{R}_i\}, \gamma \rangle$
- Framework for multi-agent decision-making

**Joint Action Space**
- Cartesian product of individual action spaces
- $\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times \ldots \times \mathcal{A}_n$
- Grows exponentially with number of agents

**Joint Policy**
- Policy that outputs actions for all agents
- $\pi(a_1, \ldots, a_n | s)$
- Used in centralized approaches

### 5.2 Environment Types

**Cooperative Environment**
- Agents share common reward signal
- Goal: Maximize team performance
- Examples: Robot teams, multi-player puzzle games

**Competitive Environment**
- Agents have conflicting objectives
- Often zero-sum: one agent's gain = another's loss
- Examples: Chess, poker, tennis

**Mixed Environment**
- Combination of cooperation and competition
- Cooperate with teammates, compete with opponents
- Examples: Soccer, team-based games

**Zero-Sum Game**
- Total rewards sum to zero: $\sum_i r_i = 0$
- Pure competition (no mutual benefit possible)
- Special case of competitive environment

### 5.3 Training Approaches

**Decentralized Learning** (Independent Learners)
- Each agent trains independently
- Treats other agents as part of environment
- Simple but suffers from non-stationarity

**Centralized Learning**
- Single policy controls all agents
- Uses joint observations and actions
- Stationary but doesn't scale well

**CTDE** (Centralized Training, Decentralized Execution)
- Training: Use centralized information
- Execution: Each agent acts on local observations
- Best of both worlds approach

**Non-Stationary Environment**
- Environment dynamics change over time
- In MARL: caused by other agents changing policies
- Violates standard RL convergence assumptions
- Main challenge of decentralized learning

### 5.4 Game Theory Concepts

**Nash Equilibrium**
- Strategy profile where no agent benefits from unilateral deviation
- $V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i, \pi_{-i}^*)$ for all $i$, all $\pi_i$
- Fundamental solution concept in game theory
- Not always unique; may not be Pareto optimal

**Best Response**
- Optimal strategy against given opponent strategies
- $\pi_i^{BR} = \arg\max_{\pi_i} V_i(\pi_i, \pi_{-i})$
- Nash equilibrium: all agents play best responses simultaneously

**Pareto Optimality**
- No agent can improve without another agent getting worse
- Efficiency criterion for cooperation
- Nash equilibrium may not be Pareto optimal (Prisoner's Dilemma)

### 5.5 Self-Play

**Self-Play**
- Training method where agent plays against past versions of itself
- Automatic curriculum: opponent difficulty scales with agent
- Used for: AlphaGo, AlphaZero, OpenAI Five, AlphaStar

**Opponent Pool**
- Collection of past policies used as opponents
- Larger pool → more diversity, more robustness
- Controlled by `window` hyperparameter

**League Training**
- Extension of self-play with multiple agent types
- Main agents, exploiters, league exploiters
- Used by AlphaStar to reach grandmaster level

### 5.6 Evaluation Metrics

**ELO Rating System**
- Relative skill rating for zero-sum games
- Updates based on game outcomes and expected performance
- Named after Arpad Elo
- Formula: $R_{\text{new}} = R_{\text{old}} + K(S - E)$

**Expected Score**
- Predicted probability of winning based on rating difference
- $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$
- Core of ELO calculations

**K-Factor**
- Maximum rating change per game
- Larger K → more volatile ratings
- Typical values: 16-32

### 5.7 Challenges in MARL

**Credit Assignment Problem**
- Difficulty determining which agent(s) contributed to outcome
- Especially challenging in cooperative settings
- Worse with delayed rewards

**Curse of Dimensionality**
- Joint spaces grow exponentially with agents
- Makes learning intractable for large teams
- Requires factorization or approximation

**Non-Stationarity**
- Environment changes as other agents learn
- Violates MDP assumptions
- Main cause of training instability

**Partial Observability**
- Agents may not observe full state
- Common in realistic scenarios
- Requires memory (RNNs, attention) or communication

### 5.8 Algorithms and Techniques

**QMIX**
- Value factorization method
- Learns individual Q-functions and mixing network
- Enforces monotonicity for decentralized execution

**MADDPG** (Multi-Agent DDPG)
- Extends DDPG to multi-agent settings
- Centralized critic, decentralized actors
- CTDE approach

**COMA** (Counterfactual Multi-Agent Policy Gradient)
- Uses counterfactual baselines for credit assignment
- Centralized critic estimates advantage
- Addresses multi-agent credit assignment

**MAPPO** (Multi-Agent PPO)
- Extends PPO to multi-agent
- Parameter sharing across agents
- Simple and effective for many cooperative tasks

### 5.9 ML-Agents Specific

**`save_steps`**
- Frequency of saving policies to opponent pool
- Affects diversity of opponent skill levels
- Typical: 50,000 steps

**`swap_steps`**
- Frequency of changing current opponent
- Affects training stability vs diversity
- Typical: 25,000 steps

**`window`**
- Size of opponent pool (how many past policies kept)
- Affects opponent diversity and memory usage
- Typical: 5-10

**`play_against_latest_model_ratio`**
- Probability of playing against most recent policy
- Balance between challenge and robustness
- Typical: 0.5

**`team_change`**
- Steps before swapping team sides
- Ensures balanced training in asymmetric games
- Typical: 200,000 steps

### 5.10 Key Equations Summary

| Concept | Equation |
|---------|----------|
| **Expected ELO Score** | $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$ |
| **ELO Update** | $R_{\text{new}} = R_{\text{old}} + K(S - E)$ |
| **Joint Action Space** | $\|\mathcal{A}\| = \prod_{i=1}^n \|\mathcal{A}_i\|$ |
| **Nash Equilibrium** | $V_i(\pi_i^*, \pi_{-i}^*) \geq V_i(\pi_i, \pi_{-i}^*)$ |
| **Team Reward (Coop)** | $r_t = \sum_{i=1}^n r_t^i$ |
| **Zero-Sum Reward** | $\sum_{i=1}^n r_t^i = 0$ |

---

## References

### Academic Papers

- **Self-Play**:
  - Silver et al. (2016), "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
  - Silver et al. (2017), "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
  - Vinyals et al. (2019), "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (AlphaStar)

- **MARL Algorithms**:
  - Rashid et al. (2018), "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"
  - Lowe et al. (2017), "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)
  - Foerster et al. (2018), "Counterfactual Multi-Agent Policy Gradients" (COMA)

- **Game Theory**:
  - Nash, J. (1950), "Equilibrium Points in N-Person Games"
  - Shoham & Leyton-Brown (2009), "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations"

### Resources

- [Unity ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents)
- [OpenAI blog on Multi-Agent](https://openai.com/blog/tags/multiagent/)
- [DeepMind AlphaStar blog](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii)
- Andrew Cohen's blog on Self-Play with ML-Agents

---

*These notes cover Unit 7: Multi-Agent Reinforcement Learning from the Hugging Face Deep RL Course. The core insight: agents learning together create richer, more robust behaviors than agents learning alone — whether through cooperation, competition, or both.*