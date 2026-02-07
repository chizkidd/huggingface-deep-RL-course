# Unit 5: Introduction to Unity ML-Agents

# Table of Contents

- [1. Introduction](#1-introduction)
  - [The Challenge of Creating Environments](#the-challenge-of-creating-environments)
  - [Game Engines as Environment Builders](#game-engines-as-environment-builders)
    - [Popular Game Engines](#popular-game-engines)
    - [Why Game Engines?](#why-game-engines)
    - [Games Made with Unity](#games-made-with-unity)
  - [Unity ML-Agents Toolkit](#unity-ml-agents-toolkit)
  - [Unit Overview](#unit-overview)
  - [The Environments](#the-environments)
    - [Environment 1: SnowballTarget](#environment-1-snowballtarget)
    - [Environment 2: Pyramids](#environment-2-pyramids)

- [2. How ML-Agents Works](#2-how-ml-agents-works)
  - [What is Unity ML-Agents?](#what-is-unity-ml-agents)
  - [The Six Components of ML-Agents](#the-six-components-of-ml-agents)
    - [Component Architecture](#component-architecture)
    - [Learning Environment](#learning-environment)
    - [Python Low-Level API](#python-low-level-api)
    - [External Communicator](#external-communicator)
    - [Python Trainers](#python-trainers)
    - [Gym Wrapper](#gym-wrapper)
    - [PettingZoo Wrapper](#pettingzoo-wrapper)
  - [Inside the Learning Component](#inside-the-learning-component)
    - [Agent Component](#agent-component)
    - [Academy](#academy)
  - [The RL Process in ML-Agents](#the-rl-process-in-ml-agents)
    - [Standard RL Loop](#standard-rl-loop)
    - [Example: Platform Game](#example-platform-game)
  - [Academy's Role in Synchronization](#academys-role-in-synchronization)
  - [Communication Flow](#communication-flow)

- [3. The SnowballTarget Environment](#3-the-snowballtarget-environment)
  - [Environment Overview](#environment-overview)
  - [The Agent's Goal](#the-agents-goal)
  - [The Cooldown System](#the-cooldown-system)
  - [The Reward Function](#the-reward-function)
    - [Reward Structure](#reward-structure)
    - [Code Implementation](#code-implementation)
  - [The Reward Engineering Problem](#the-reward-engineering-problem)
  - [The Observation Space](#the-observation-space)
    - [Raycasts: The Agent's Sensors](#raycasts-the-agents-sensors)
    - [Observation Components](#observation-components)
    - [Code Structure](#code-structure)
  - [The Action Space](#the-action-space)
    - [Discrete Action Space](#discrete-action-space)
  - [Training Considerations](#training-considerations)
    - [Exploration-Exploitation Trade-off](#exploration-exploitation-trade-off)
    - [Key Hyperparameters](#key-hyperparameters)
  - [Success Metrics](#success-metrics)

- [4. The Pyramids Environment](#4-the-pyramids-environment)
  - [Environment Overview](#environment-overview-1)
  - [The Agent's Goal](#the-agents-goal-1)
  - [The Reward Function](#the-reward-function-1)
    - [Extrinsic Rewards](#extrinsic-rewards)
    - [Code Implementation](#code-implementation-1)
    - [Intrinsic Rewards: Curiosity](#intrinsic-rewards-curiosity)
  - [The Observation Space](#the-observation-space-1)
    - [Raycast Configuration](#raycast-configuration)
    - [Additional Observations](#additional-observations)
    - [Code Structure](#code-structure-1)
  - [The Action Space](#the-action-space-1)
    - [Discrete Actions](#discrete-actions)
    - [Multi-Discrete Action Space](#multi-discrete-action-space)
  - [Training with Curiosity](#training-with-curiosity)
    - [Why Curiosity is Essential](#why-curiosity-is-essential)
    - [Hyperparameters for Pyramids](#hyperparameters-for-pyramids)
  - [Learning Progression](#learning-progression)

- [5. Curiosity in Deep Reinforcement Learning](#5-curiosity-in-deep-reinforcement-learning)
  - [The Two Major Problems in Modern RL](#the-two-major-problems-in-modern-rl)
    - [Problem 1: Sparse Rewards](#problem-1-sparse-rewards)
    - [Problem 2: Hand-Crafted Reward Functions](#problem-2-hand-crafted-reward-functions)
  - [So What is Curiosity?](#so-what-is-curiosity)
  - [Curiosity Through Next-State Prediction](#curiosity-through-next-state-prediction)
  - [Feature-Based Curiosity (ICM)](#feature-based-curiosity-icm)
  - [Random Network Distillation (RND)](#random-network-distillation-rnd)
  - [Practical Considerations for Curiosity](#practical-considerations-for-curiosity)
  - [Mathematical Summary](#mathematical-summary)

- [6. Glossary](#6-glossary)
  - [Core Concepts](#core-concepts)
  - [Environment Components](#environment-components)
  - [Reward Concepts](#reward-concepts)
  - [Curiosity Methods](#curiosity-methods)
  - [Training Algorithms](#training-algorithms)
  - [ML-Agents Specific](#ml-agents-specific)
  - [Mathematical Notation](#mathematical-notation)



## Appendix

- [Additional Concepts](#additional-concepts)
  - [Multi-Agent Training](#multi-agent-training)
  - [Self-Play](#self-play)
  - [Curriculum Learning](#curriculum-learning)
  - [Transfer Learning](#transfer-learning)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Best Practices](#best-practices)
- [Summary](#summary)
- [References and Resources](#references-and-resources)


---

## 1. Introduction

### The Challenge of Creating Environments

One of the fundamental challenges in Reinforcement Learning is **creating realistic and complex environments** for training agents. Fortunately, we can leverage game engines to solve this problem.

### Game Engines as Environment Builders

**Popular Game Engines**:
- [Unity](https://unity.com/)
- [Godot](https://godotengine.org/)
- [Unreal Engine](https://www.unrealengine.com/)

**Why Game Engines?**
- Provide built-in physics systems
- Offer 2D/3D rendering capabilities
- Handle collision detection
- Manage complex interactions
- Are optimized for performance

**Games Made with Unity**:
- Firewatch
- Cuphead
- Cities: Skylines
- Hearthstone
- Monument Valley

### Unity ML-Agents Toolkit

**Definition**: A plugin for the Unity Game Engine that allows us to:
- Use Unity as an environment builder
- Train RL agents in Unity-created environments
- Use pre-made environments for training

**Developer**: Unity Technologies

**Key Feature**: Bridges game development and machine learning

### Unit Overview

In this unit, we will:

1. **Learn Unity ML-Agents** (no Unity knowledge required!)
2. **Train two agents**:
   - **Julien the Bear**: Learn to shoot snowballs at spawning targets
   - **Pyramid Agent**: Navigate environment, find button, spawn pyramid, and reach gold brick

3. **Use Advanced Techniques**:
   - Intrinsic rewards
   - Curiosity-driven learning
   - Exploration strategies

4. **Share Models**: Push trained agents to Hugging Face Hub

5. **Preparation**: Get ready for AI vs. AI multi-agent challenges

### The Environments

**Environment 1: SnowballTarget**
- Agent: Julien the Bear 🐻
- Task: Hit targets with snowballs
- Constraint: Cooldown system between shots
- Goal: Maximize targets hit in limited time

**Environment 2: Pyramids**
- Task: Multi-step sequential goal
  1. Press button to spawn pyramid
  2. Navigate to pyramid
  3. Knock pyramid over
  4. Reach gold brick at top
- Feature: Uses curiosity for exploration

---

## 2. How ML-Agents Works

### What is Unity ML-Agents?

**Official Definition**: A toolkit for the Unity game engine that enables:
- Creating custom RL environments using Unity
- Using pre-made environments for training
- Training agents with state-of-the-art RL algorithms

**Key Capability**: Allows game developers and researchers to train intelligent agents in realistic 3D environments

### The Six Components of ML-Agents

#### Component Architecture

```
┌─────────────────────────────────────────────────┐
│          Unity ML-Agents System                 │
├─────────────────────────────────────────────────┤
│ 1. Learning Environment (Unity Scene)           │
│    └─ Game characters, objects, physics         │
├─────────────────────────────────────────────────┤
│ 2. Python Low-Level API                         │
│    └─ Interface for environment manipulation    │
├─────────────────────────────────────────────────┤
│ 3. External Communicator                        │
│    └─ Bridge between C# (Unity) and Python      │
├─────────────────────────────────────────────────┤
│ 4. Python Trainers                              │
│    └─ RL algorithms (PPO, SAC, etc.)           │
├─────────────────────────────────────────────────┤
│ 5. Gym Wrapper                                  │
│    └─ Single-agent RL environment wrapper       │
├─────────────────────────────────────────────────┤
│ 6. PettingZoo Wrapper                           │
│    └─ Multi-agent RL environment wrapper        │
└─────────────────────────────────────────────────┘
```

#### 1. Learning Environment

**Components**:
- Unity scene (the environment)
- Environment elements (game characters, objects, terrain)
- Physics simulation
- Visual rendering

**Built with**: C# and Unity Engine

**Purpose**: Provides the simulated world where agents operate

#### 2. Python Low-Level API

**Functionality**:
- Low-level Python interface for environment interaction
- Methods to manipulate environment state
- Controls for stepping through simulation
- API for launching training

**Language**: Python

**Usage**: Called by training scripts to control environment

#### 3. External Communicator

**Purpose**: Bridge/connector between different technologies

**Connection**:
```
Unity (C#) ←→ External Communicator ←→ Python API
```

**Function**: Enables Unity and Python to communicate seamlessly

#### 4. Python Trainers

**Available Algorithms**:
- **PPO** (Proximal Policy Optimization) - Primary algorithm
- **SAC** (Soft Actor-Critic)
- Other PyTorch-based RL algorithms

**Framework**: Built with PyTorch

**Purpose**: Implement RL training algorithms

#### 5. Gym Wrapper

**Purpose**: Encapsulate RL environment in OpenAI Gym format

**Benefits**:
- Standardized interface
- Compatible with Gym-based tools
- Familiar API for researchers

**Use Case**: Single-agent environments

#### 6. PettingZoo Wrapper

**Purpose**: Multi-agent version of Gym wrapper

**Features**:
- Handles multiple agents
- Supports competitive/cooperative scenarios
- Standard multi-agent interface

**Use Case**: Multi-agent environments

### Inside the Learning Component

The Learning Component contains two critical elements:

#### 1. Agent Component

**Definition**: The actor in the scene that we train

**Key Aspects**:
- Receives observations from environment
- Takes actions based on policy
- Receives rewards
- Policy called the "Brain"

**Training Process**: Optimize the agent's policy (Brain)

**Policy Function**:
```
π(a|s) = Probability of taking action a given state s
```

#### 2. Academy

**Definition**: Orchestrator of agents and decision-making

**Metaphor**: Teacher handling Python API requests

**Responsibilities**:
- Synchronize all agents
- Send orders to agents
- Ensure agents stay in sync
- Manage episode lifecycle

**Commands Issued by Academy**:
1. Collect Observations
2. Select action using policy
3. Take the action
4. Reset if max steps reached or episode done

### The RL Process in ML-Agents

#### Standard RL Loop

The RL process can be modeled as a loop:

```
State (S) → Agent → Action (A) → Environment → Reward (R) + Next State (S')
    ↑                                                              ↓
    └──────────────────────────────────────────────────────────────┘
```

**Mathematical Formulation**:

**At timestep t**:
1. Agent in state $S_t$
2. Selects action $A_t$ using policy $\pi(a|s)$
3. Environment transitions to $S_{t+1}$
4. Agent receives reward $R_{t+1}$

**Objective**: Maximize expected cumulative reward

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Where:
- $G_t$ = Return (cumulative reward) from timestep $t$
- $\gamma$ = Discount factor (typically 0.99)
- $R_{t+k+1}$ = Reward at future timestep $t+k+1$

#### Example: Platform Game

**Sequence**:

**Timestep 0**:
```
State S₀: First frame of game
Action A₀: Move right
```

**Timestep 1**:
```
Environment → New state S₁: New frame
Reward R₁: +1 (not dead)
```

**Output Sequence**:
```
(S₀, A₀, R₁, S₁) → (S₁, A₁, R₂, S₂) → (S₂, A₂, R₃, S₃) → ...
```

**Goal**: Maximize $E[G_t]$ (expected cumulative reward)

### Academy's Role in Synchronization

**Academy Workflow**:

```
For each timestep:
    Academy sends commands:
        1. "Collect Observations" → All agents observe environment
        2. "Select Actions" → All agents use policy to choose actions
        3. "Take Actions" → All agents execute actions
        4. "Check Termination" → Reset agents if needed
    
    Ensure all agents synchronized
```

**Benefits of Synchronization**:
- Consistent training across multiple agents
- Proper multi-agent interaction
- Stable learning dynamics
- Efficient parallel training

### Communication Flow

```
┌─────────────────┐
│  Unity Engine   │
│  (C# Scripts)   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   External      │
│  Communicator   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Python API     │
│  (mlagents)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  RL Trainer     │
│  (PPO/SAC)      │
└─────────────────┘
```

**Data Flow**:
1. **Unity → Python**: Observations, rewards, done flags
2. **Python → Unity**: Actions to execute
3. **Bidirectional**: Configuration, hyperparameters

---

## 3. The SnowballTarget Environment

### Environment Overview

**Agent**: Julien the Bear 🐻

**Task**: Hit spawning targets with snowballs

**Challenge**: Limited time (1000 timesteps) and cooldown system

**Created by**: Hugging Face using assets from Kay Lousberg

**Difficulty**: Moderate - requires aim and timing

### The Agent's Goal

**Primary Objective**: Hit as many targets as possible within time limit

**Required Skills**:
1. **Position**: Place itself correctly relative to target
2. **Aim**: Orient toward target
3. **Shoot**: Fire snowball at appropriate time
4. **Timing**: Manage cooldown between shots

**Time Constraint**: 1000 timesteps per episode

### The Cooldown System

**Purpose**: Prevent "snowball spamming" (shooting every timestep)

**Mechanism**:
- After shooting, agent must wait 0.5 seconds
- Cannot shoot during cooldown period
- Adds strategic element to gameplay

**Impact on Strategy**:
- Agent must make shots count
- Encourages better positioning
- Promotes accurate aiming
- Prevents brute-force approaches

**Cooldown Duration**: 0.5 seconds

### The Reward Function

#### Reward Structure

**Simple Reward System**:
```
R(hit) = +1   (snowball hits target)
R(miss) = 0   (snowball misses or no action)
```

**Mathematical Formulation**:

$$R_t = \begin{cases}
+1 & \text{if snowball hits target at timestep } t \\
0 & \text{otherwise}
\end{cases}$$

**Cumulative Reward**:

$$G = \sum_{t=1}^{T} R_t = \text{Total number of targets hit}$$

Where $T$ = episode length (1000 timesteps)

#### Code Implementation

```csharp
// When snowball hits target
public void OnTargetHit()
{
    AddReward(1.0f);
    // Spawn new target
    RespawnTarget();
}

// No penalty for missing
```

### The Reward Engineering Problem

#### What is Reward Engineering?

**Definition**: Creating overly complex reward functions to force specific agent behavior

**Temptation**: Add penalties/bonuses for everything
- Penalty for taking too long
- Bonus for moving toward target
- Penalty for wasting snowballs
- Bonus for hitting quickly

**Why This is Problematic**:

1. **Miss Interesting Strategies**
   - Agent might find creative solutions
   - Complex rewards constrain exploration
   - Simple rewards allow emergent behavior

2. **Harder to Debug**
   - Multiple reward components interact
   - Difficult to identify issues
   - Balancing becomes complex

3. **May Not Generalize**
   - Overfitted to specific behaviors
   - Doesn't adapt to variations
   - Fragile to environment changes

**Best Practice**: Start simple, add complexity only if needed

**Philosophy**:
```
Simple Reward + Smart Algorithm > Complex Reward + Basic Algorithm
```

### The Observation Space

#### Raycasts: The Agent's Sensors

**What are Raycasts?**
- Like laser beams detecting objects
- Return information about what they hit
- Provide distance and object type
- Similar to sonar or lidar

**Benefits vs. Vision**:
- More efficient than camera images
- Structured, low-dimensional input
- Faster to process
- Easier to train with

**Raycast Information**:
```
For each raycast:
    - Distance to object (normalized)
    - Object type (one-hot encoded)
        - Target
        - Wall
        - Nothing
```

#### Observation Components

**1. Multiple Raycast Sets**:
```
Raycasts arranged in patterns:
    - Forward-facing rays (detect targets ahead)
    - Side rays (peripheral vision)
    - Different angles (wide field of view)
```

**2. "Can I Shoot?" Boolean**:
```
can_shoot = {
    True  if cooldown_timer == 0
    False if cooldown_timer > 0
}
```

**Total Observation Vector**:
```
Observation = [
    raycast_data[0],      # First raycast set
    raycast_data[1],      # Second raycast set
    ...
    raycast_data[n],      # nth raycast set
    can_shoot             # Boolean (0 or 1)
]
```

#### Code Structure

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    // Add raycast observations
    // Each raycast detects objects and distances
    
    // Add cooldown status
    sensor.AddObservation(canShoot);
}
```

**Observation Dimension**: Variable (depends on number of raycasts + 1)

### The Action Space

#### Discrete Action Space

**Action Type**: Discrete (finite set of distinct actions)

**Available Actions**:

| Action ID | Action Name | Description |
|-----------|-------------|-------------|
| 0 | Nothing | No action, idle |
| 1 | Move Forward | Move in forward direction |
| 2 | Move Backward | Move in backward direction |
| 3 | Rotate Left | Turn counterclockwise |
| 4 | Rotate Right | Turn clockwise |
| 5 | Shoot | Fire snowball (if cooldown ready) |

**Action Space Size**: 6 discrete actions

**Mathematical Representation**:

$$\mathcal{A} = \{a_0, a_1, a_2, a_3, a_4, a_5\}$$

**Policy Output**:

$$\pi(a|s) = \text{Probability distribution over actions}$$

Example output:
```
π(Nothing|s) = 0.05
π(Forward|s) = 0.30
π(Backward|s) = 0.02
π(Rotate Left|s) = 0.08
π(Rotate Right|s) = 0.10
π(Shoot|s) = 0.45     ← Highest probability
```

Agent selects action with highest probability (or samples from distribution)

### Training Considerations

#### Exploration-Exploitation Trade-off

**Challenge**: Balance between:
- **Exploration**: Try different shooting positions and angles
- **Exploitation**: Use known good positions

**Solution**: Entropy bonus in PPO encourages exploration

**Entropy Regularization**:

$$J_{total} = J_{policy} + \beta H(\pi)$$

Where:
- $J_{policy}$ = Policy gradient objective
- $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$ = Policy entropy
- $\beta$ = Entropy coefficient (e.g., 0.01)

High entropy = more random actions = more exploration

#### Key Hyperparameters

**For SnowballTarget**:

```yaml
behaviors:
  SnowballTarget:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 5.0e-3          # Entropy coefficient
      epsilon: 0.2          # PPO clip range
      lambd: 0.95           # GAE lambda
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 128
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99         # Discount factor
        strength: 1.0
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
```

### Success Metrics

**Performance Indicators**:

1. **Mean Cumulative Reward**: Average targets hit per episode
2. **Episode Length**: How long agent survives
3. **Success Rate**: Percentage of episodes hitting > N targets
4. **Shooting Accuracy**: Hits / Total shots

**Training Progress**:
```
Early Training:
    - Random shooting
    - Poor positioning
    - Low accuracy
    - ~0-5 targets hit

Mid Training:
    - Basic positioning
    - Some target tracking
    - Moderate accuracy
    - ~10-20 targets hit

Late Training:
    - Optimal positioning
    - Good target tracking
    - High accuracy
    - ~30-50+ targets hit
```

---

## 4. The Pyramids Environment

### Environment Overview

**Task Complexity**: Multi-step sequential goal

**Objective**: Obtain gold brick at top of pyramid

**Challenge**: Requires exploration of complex action sequence

**Reward Structure**: Sparse extrinsic + dense intrinsic (curiosity)

### The Agent's Goal

**Multi-Step Objective**:

1. **Find and Press Button** → Spawns pyramid
2. **Navigate to Pyramid** → Locate spawned structure
3. **Knock Over Pyramid** → Topple the structure
4. **Reach Gold Brick** → Move to brick at top

**Sequential Nature**:
```
Press Button → Pyramid Spawns → Navigate → Knock Over → Reach Gold
     ↓              ↓              ↓          ↓            ↓
   +Step 1      +Step 2        +Step 3    +Step 4      +Goal
```

**Difficulty**: Must learn correct action sequence through trial and error

### The Reward Function

#### Extrinsic Rewards

**Reward Structure**:

$$R_{extrinsic}(s, a) = \begin{cases}
+1 & \text{if button pressed (pyramid spawned)} \\
+1 & \text{if pyramid knocked over} \\
+1 & \text{if gold brick reached} \\
0 & \text{otherwise}
\end{cases}$$

**Characteristics**:
- **Sparse**: Only 3 reward events in entire episode
- **Sequential**: Must achieve in order
- **Challenging**: Long gaps between rewards

#### Code Implementation

```csharp
// Button pressed
if (buttonPressed && !pyramidSpawned)
{
    AddReward(1.0f);
    SpawnPyramid();
}

// Pyramid knocked over
if (pyramidKnockedOver && !previouslyKnocked)
{
    AddReward(1.0f);
}

// Gold brick reached
if (reachedGoldBrick)
{
    AddReward(1.0f);
    EndEpisode();
}
```

#### Intrinsic Rewards: Curiosity

**Purpose**: Encourage exploration in sparse reward environment

**Combination**:

$$R_{total} = R_{extrinsic} + \beta \cdot R_{intrinsic}$$

Where:
- $R_{total}$ = Combined reward used for training
- $R_{extrinsic}$ = Environment-provided reward
- $R_{intrinsic}$ = Curiosity-driven reward
- $\beta$ = Intrinsic reward strength (e.g., 0.01)

**Why Necessary?**
- Extrinsic rewards too sparse
- Agent needs guidance for exploration
- Curiosity fills gaps between milestones

### The Observation Space

#### Raycast Configuration

**Purpose**: Detect objects in environment

**148 Raycasts** detecting:
- Switch (button)
- Pyramid bricks
- Golden brick
- Walls
- Empty space

**Raycast Arrangement**:
```
       ↑ ↑ ↑ ↑ ↑        Forward rays
      ↑ ↑ ↑ ↑ ↑
     ← → ← → ← →        Side rays
    ↙ ↙   ↘ ↘          Diagonal rays
```

**Each Raycast Returns**:
1. **Hit**: Boolean (did it hit something?)
2. **Distance**: Float (how far away?)
3. **Object Type**: One-hot encoding
   - Switch: [1, 0, 0, 0, 0]
   - Brick: [0, 1, 0, 0, 0]
   - Gold: [0, 0, 1, 0, 0]
   - Wall: [0, 0, 0, 1, 0]
   - Nothing: [0, 0, 0, 0, 1]

#### Additional Observations

**1. Switch State**:
```csharp
bool switchActivated; // Has button been pressed?
sensor.AddObservation(switchActivated);
```

**2. Agent Velocity**:
```csharp
Vector3 velocity;  // Agent's speed in x, y, z
sensor.AddObservation(velocity.x);
sensor.AddObservation(velocity.y);
sensor.AddObservation(velocity.z);
```

**Total Observation Vector**:
```
Observation = [
    raycast_data[0..147],    # 148 raycasts × features per ray
    switch_state,            # Boolean
    velocity_x,              # Float
    velocity_y,              # Float  
    velocity_z               # Float
]
```

#### Code Structure

```csharp
public override void CollectObservations(VectorSensor sensor)
{
    // 148 raycasts automatically added
    
    // Switch state
    sensor.AddObservation(switchActivated);
    
    // Velocity
    Vector3 localVelocity = transform.InverseTransformDirection(rb.velocity);
    sensor.AddObservation(localVelocity.x);
    sensor.AddObservation(localVelocity.y);
    sensor.AddObservation(localVelocity.z);
}
```

### The Action Space

#### Discrete Actions

**Action Type**: Discrete with 4 possible actions

**Available Actions**:

| Branch | Action ID | Action Name | Description |
|--------|-----------|-------------|-------------|
| 0 | 0 | Nothing | No movement |
| 0 | 1 | Forward | Move forward |
| 0 | 2 | Backward | Move backward |
| 1 | 0 | Nothing | No rotation |
| 1 | 1 | Rotate Left | Turn left |
| 1 | 2 | Rotate Right | Turn right |

**Multi-Discrete Action Space**:

$$\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2$$

Where:
- $\mathcal{A}_1 = \{Nothing, Forward, Backward\}$ (Movement)
- $\mathcal{A}_2 = \{Nothing, Left, Right\}$ (Rotation)

**Total Combinations**: 3 × 3 = 9 possible action combinations

**Example Action Combinations**:
- (Forward, Left): Move forward while turning left
- (Backward, Right): Move backward while turning right
- (Forward, Nothing): Move straight forward
- (Nothing, Nothing): Stay still

**Policy Output**:

$$\pi(a_1, a_2 | s) = \pi_1(a_1|s) \cdot \pi_2(a_2|s)$$

Assuming independence between action branches

### Training with Curiosity

#### Why Curiosity is Essential

**The Sparse Reward Problem**:
```
Episode Timeline (1000 steps):
0────────→200────────→500────────→800────────→1000
│          │           │           │            │
Start    Maybe      Maybe       Maybe        End
         Button     Pyramid     Gold
        (+1?)       (+1?)       (+1?)

Most steps: Reward = 0
```

**Without Curiosity**:
- Agent wanders randomly
- Rarely discovers button by chance
- Takes extremely long to learn
- May never solve task

**With Curiosity**:
- Rewarded for exploring new areas
- Naturally discovers button
- Learns faster through exploration
- Consistently solves task

#### Hyperparameters for Pyramids

```yaml
behaviors:
  Pyramids:
    trainer_type: ppo
    hyperparameters:
      batch_size: 128
      buffer_size: 2048
      learning_rate: 3.0e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: false
      hidden_units: 256       # Larger network
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
      curiosity:              # Intrinsic motivation
        gamma: 0.99
        strength: 0.01        # β in formula
        encoding_size: 256
        learning_rate: 3.0e-4
    max_steps: 1000000        # More steps needed
    time_horizon: 128
    summary_freq: 10000
```

### Learning Progression

**Phase 1: Random Exploration** (0-100k steps)
- Random movements
- Occasionally hits button by luck
- Curiosity drives exploration
- No consistent strategy

**Phase 2: Button Discovery** (100k-300k steps)
- Learns to find button
- Pyramid spawning becomes consistent
- Begins exploring around pyramid
- Still struggles with knocking it over

**Phase 3: Pyramid Manipulation** (300k-600k steps)
- Learns to knock pyramid
- Discovers gold brick
- Refines navigation
- Increasing success rate

**Phase 4: Mastery** (600k+ steps)
- Efficient button pressing
- Direct navigation to pyramid
- Reliable pyramid toppling
- Consistent gold brick retrieval

---

## 5. Curiosity in Deep Reinforcement Learning

### The Two Major Problems in Modern RL

#### Problem 1: Sparse Rewards

**Definition**: Most rewards contain no information (are zero)

**Impact on Learning**:
- RL based on reward hypothesis: all goals = maximizing rewards
- Rewards act as feedback for agents
- **No reward = No learning signal**
- Agent can't determine if actions are good or bad

**Mathematical Formulation**:

In sparse reward environment:

$$R_t = \begin{cases}
+1 & \text{if goal achieved (rare)} \\
0 & \text{otherwise (most of the time)}
\end{cases}$$

**Expected reward in episode**:

$$E[G] = E\left[\sum_{t=0}^{T} \gamma^t R_t\right] \approx 0$$

Most of the time, as most $R_t = 0$

**Example: Vizdoom "DoomMyWayHome"**

**Environment**:
- Agent spawns far from goal (vest)
- Reward: +1 only when vest found
- Everything else: Reward = 0

**Problem**:
```
Episode (500 steps):
Step 0-498: Reward = 0, 0, 0, 0, ... (no feedback)
Step 499: Found vest! Reward = +1 (finally!)

Agent doesn't know:
- Were steps 0-498 helpful?
- Which direction is correct?
- What actions lead toward goal?
```

**Consequence**: Extremely slow learning, agent may wander aimlessly

#### Problem 2: Hand-Crafted Reward Functions

**The Issue**: Every environment needs human-designed reward function

**Challenges**:

1. **Domain Expertise Required**
   - Need to understand task deeply
   - Know what behaviors to encourage
   - Predict unintended consequences

2. **Scaling Problems**
   - Each new environment needs new reward
   - Complex environments = complex rewards
   - Time-consuming to design and test

3. **Reward Hacking**
   - Agent finds loopholes
   - Maximizes reward in unexpected ways
   - Doesn't actually solve intended task

**Example Reward Hacking**:
```
Intended: Agent should clean room
Reward: +1 per object put away

Agent's solution:
- Picks up object
- Puts it away (+1)
- Takes it out again
- Puts it away (+1)
- Repeat forever
```

**Question**: How to scale to complex, diverse environments?

### So What is Curiosity?

#### Core Concept

**Definition**: An intrinsic reward mechanism generated by the agent itself

**Key Idea**: Agent becomes its own teacher
- Acts as student (learner)
- Acts as feedback master (reward generator)
- Self-supervised learning

**Inspiration**: Human psychology
- Humans naturally explore novel environments
- Intrinsic desire to discover new things
- Curiosity drives learning even without external rewards

#### Intrinsic vs Extrinsic Rewards

**Extrinsic Rewards**:
- Provided by environment
- Designed by humans
- Task-specific
- Sparse in difficult environments

$$R_{extrinsic}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

**Intrinsic Rewards**:
- Generated by agent itself
- Based on novelty/surprise
- Task-agnostic (general)
- Dense (provided frequently)

$$R_{intrinsic}: \mathcal{S} \times \mathcal{A} \times \mathcal{S}' \rightarrow \mathbb{R}$$

**Combined Reward**:

$$R_{total} = R_{extrinsic} + \beta \cdot R_{intrinsic}$$

Where $\beta$ controls importance of intrinsic motivation (typically 0.001-0.1)

### Curiosity Through Next-State Prediction

#### The Classical Approach

**Core Principle**: Measure curiosity as prediction error

**Mechanism**:
1. Agent predicts next state given current state and action
2. Compares prediction to actual next state
3. Prediction error = Curiosity reward

**Mathematical Formulation**:

**Forward Model**: Predicts next state

$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

Where:
- $\hat{s}_{t+1}$ = Predicted next state
- $f_\theta$ = Forward model (neural network)
- $s_t$ = Current state
- $a_t$ = Action taken
- $\theta$ = Model parameters

**Prediction Error** (Curiosity Reward):

$$r_{curiosity}(s_t, a_t, s_{t+1}) = \|\hat{s}_{t+1} - s_{t+1}\|^2$$

$r_{curiosity}(s_t, a_t, s_{t+1}) = \|f_\theta(s_t, a_t) - s_{t+1}\|^2$

**Intuition**:
- High error = Novel/unexpected state = High curiosity
- Low error = Familiar state = Low curiosity

#### Why This Works

**Exploration Mechanism**:

$\pi^* = \arg\max_\pi E\left[\sum_{t=0}^{\infty} \gamma^t (R_{ext}(s_t, a_t) + \beta \cdot r_{curiosity}(s_t, a_t, s_{t+1}))\right]$

**Behavior**:

**In Familiar Areas**:
- Agent can predict next state accurately
- $\|f_\theta(s_t, a_t) - s_{t+1}\|^2 \approx 0$
- Low intrinsic reward
- Agent moves on to explore elsewhere

**In Novel Areas**:
- Agent cannot predict next state well
- $\|f_\theta(s_t, a_t) - s_{t+1}\|^2 >> 0$
- High intrinsic reward
- Agent encouraged to stay and explore

**Learning Dynamics**:
```
New Area → High Prediction Error → High Curiosity → Agent Explores
   ↓
Agent Spends Time → Learns to Predict → Low Prediction Error
   ↓
Low Curiosity → Agent Moves to New Area → Cycle Repeats
```

#### Training the Forward Model

**Loss Function**:

$\mathcal{L}_{forward} = E_{(s_t, a_t, s_{t+1}) \sim D}\left[\|f_\theta(s_t, a_t) - s_{t+1}\|^2\right]$

**Update Rule**:

$\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_{forward}$

Where:
- $D$ = Replay buffer of experiences
- $\alpha$ = Learning rate for forward model

**Dual Learning Process**:
1. **Policy Network**: Learns to maximize total reward
2. **Forward Model**: Learns to predict next states

Both trained simultaneously during RL training

### Feature-Based Curiosity (ICM)

#### The Problem with Raw States

**Issue**: Predicting raw pixel values is hard and unnecessary

**Example**:
- Predicting exact pixel colors in next frame
- Includes irrelevant details (clouds moving, trees swaying)
- High-dimensional, noisy predictions

**Solution**: Use learned feature representations

#### Intrinsic Curiosity Module (ICM)

**Architecture**:

```
State s_t ─────────→ Feature Network φ ─────→ Feature f_t
                            ↓
Action a_t ──────────┐     ↓
                     ↓     ↓
              Forward Model ─→ Predicted Feature f̂_{t+1}
                     
State s_{t+1} ──→ Feature Network φ ─────→ Actual Feature f_{t+1}
                     
Prediction Error: ||f̂_{t+1} - f_{t+1}||² = Curiosity Reward
```

**Feature Network**: 

$f_t = \phi(s_t)$

Learns to extract relevant features from state

**Forward Model in Feature Space**:

$\hat{f}_{t+1} = g_\psi(f_t, a_t)$

Where:
- $\phi$ = Feature encoder network
- $g_\psi$ = Forward dynamics model
- $f_t$ = Feature representation of state

**Curiosity Reward**:

$r_{curiosity} = \frac{\eta}{2} \|\hat{f}_{t+1} - f_{t+1}\|^2 = \frac{\eta}{2} \|g_\psi(\phi(s_t), a_t) - \phi(s_{t+1})\|^2$

Where $\eta$ is scaling factor

#### Inverse Model (Self-Supervised Learning)

**Purpose**: Train feature network to encode action-relevant information

**Inverse Model**: Predicts action from state transition

$\hat{a}_t = h_\omega(f_t, f_{t+1})$

Where $h_\omega$ predicts action given current and next state features

**Inverse Model Loss**:

$\mathcal{L}_{inverse} = -\log P(a_t | s_t, s_{t+1})$

For discrete actions (cross-entropy loss)

$\mathcal{L}_{inverse} = \|h_\omega(\phi(s_t), \phi(s_{t+1})) - a_t\|^2$

For continuous actions (MSE loss)

**Why This Helps**:
- Forces features to encode action-relevant information
- Ignores irrelevant environmental details
- Makes predictions more meaningful

#### Complete ICM Loss

**Total Loss**:

$\mathcal{L}_{ICM} = (1-\beta_{ICM}) \mathcal{L}_{inverse} + \beta_{ICM} \mathcal{L}_{forward}$

Where:
- $\mathcal{L}_{inverse}$ = Inverse model loss (action prediction)
- $\mathcal{L}_{forward}$ = Forward model loss (next feature prediction)
- $\beta_{ICM}$ = Weight balancing two objectives (typically 0.2)

**Simplified Derivation**:

1. **Extract features**: $f_t = \phi(s_t)$, $f_{t+1} = \phi(s_{t+1})$

2. **Predict action (inverse model)**: $\hat{a}_t = h_\omega(f_t, f_{t+1})$

3. **Compute inverse loss**: $\mathcal{L}_{inv} = \|a_t - \hat{a}_t\|^2$

4. **Predict next feature (forward model)**: $\hat{f}_{t+1} = g_\psi(f_t, a_t)$

5. **Compute forward loss**: $\mathcal{L}_{fwd} = \|f_{t+1} - \hat{f}_{t+1}\|^2$

6. **Curiosity reward**: $r_{curiosity} = \eta \cdot \mathcal{L}_{fwd}$

7. **Update networks**: Minimize $\mathcal{L}_{ICM}$

### Random Network Distillation (RND)

**Note**: This is the method used by ML-Agents

#### Core Idea

**Motivation**: Simpler than ICM, often more effective

**Key Insight**: Measure novelty by how well agent can predict random network outputs

#### RND Architecture

**Two Networks**:

1. **Target Network** (random, fixed):
   $f_{target}: \mathcal{S} \rightarrow \mathbb{R}^k$
   - Randomly initialized
   - **Never trained** (parameters frozen)
   - Maps states to random feature vectors

2. **Predictor Network** (trained):
   $\hat{f}_{predictor}: \mathcal{S} \rightarrow \mathbb{R}^k$
   - Tries to predict target network output
   - Trained on visited states
   - Becomes better at familiar states

**Prediction Error (Curiosity)**:

$r_{curiosity}(s) = \|\hat{f}_{predictor}(s) - f_{target}(s)\|^2$

**Training Predictor**:

$\mathcal{L}_{RND} = E_{s \sim D}\left[\|\hat{f}_{predictor}(s) - f_{target}(s)\|^2\right]$

Where $D$ = visited states

#### Why RND Works

**Intuition**:

**Visited States** (Familiar):
- Predictor has seen many times
- Learned to approximate target network
- $\|\hat{f}_{predictor}(s) - f_{target}(s)\|^2$ is small
- Low curiosity reward

**Novel States** (Unfamiliar):
- Predictor hasn't seen before
- Cannot predict random target network
- $\|\hat{f}_{predictor}(s) - f_{target}(s)\|^2$ is large
- High curiosity reward

**Key Property**: Random target network creates consistent but unpredictable outputs for novel states

#### RND vs ICM Comparison

| Aspect | ICM | RND |
|--------|-----|-----|
| **Complexity** | 3 networks (feature, forward, inverse) | 2 networks (target, predictor) |
| **Training** | Trains all 3 networks | Trains only predictor |
| **State Dependency** | Depends on state transitions | Depends only on states |
| **Action Dependency** | Uses actions explicitly | Action-independent |
| **Computational Cost** | Higher | Lower |
| **Performance** | Good | Often better |
| **Used in ML-Agents** | No | Yes |

### Practical Considerations for Curiosity

#### Balancing Extrinsic and Intrinsic Rewards

**The β Parameter**:

$R_{total} = R_{extrinsic} + \beta \cdot R_{intrinsic}$

**Choosing β**:
- **Too small** (β < 0.001): Curiosity has little effect
- **Too large** (β > 0.1): Agent ignores extrinsic rewards
- **Typical range**: 0.001 - 0.05

**Rule of Thumb**:
- Sparse rewards → Higher β (0.01 - 0.05)
- Dense rewards → Lower β (0.001 - 0.01)

#### Normalization

**Challenge**: Curiosity rewards can have different scales than extrinsic rewards

**Solution**: Normalize intrinsic rewards

$r_{norm} = \frac{r_{curiosity} - \mu_{curiosity}}{\sigma_{curiosity}}$

Where:
- $\mu_{curiosity}$ = Running mean of curiosity rewards
- $\sigma_{curiosity}$ = Running std of curiosity rewards

**ML-Agents Implementation**: Automatic normalization in curiosity module

#### When to Use Curiosity

**Good Use Cases**:
- Sparse reward environments
- Complex exploration required
- Long episode horizons
- Multi-step sequential tasks

**Not Needed**:
- Dense reward environments
- Simple exploration (random is enough)
- Short episodes
- Well-shaped reward functions

**Example Environments**:

**Use Curiosity**:
- Pyramids (sparse rewards, exploration needed)
- Montezuma's Revenge (extremely sparse)
- Procedurally generated environments

**Don't Need Curiosity**:
- SnowballTarget (rewards frequent)
- CartPole (simple, dense rewards)
- Most continuous control tasks

### Mathematical Summary

**Complete Reward Function**:

$R_{total}(s_t, a_t, s_{t+1}) = R_{extrinsic}(s_t, a_t) + \beta \cdot R_{intrinsic}(s_t, a_t, s_{t+1})$

**For ICM**:

$R_{intrinsic} = \frac{\eta}{2} \|g_\psi(\phi(s_t), a_t) - \phi(s_{t+1})\|^2$

**For RND**:

$R_{intrinsic} = \|\hat{f}_{predictor}(s_{t+1}) - f_{target}(s_{t+1})\|^2$

**Policy Optimization Objective**:

$J(\theta) = E_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R_{total}(s_t, a_t, s_{t+1})\right]$

Where:
- $\theta$ = Policy parameters
- $\tau$ = Trajectory
- $\gamma$ = Discount factor

---

## 6. Glossary

### Core Concepts

**Unity ML-Agents Toolkit**
- Plugin for Unity game engine
- Enables creating RL environments using Unity
- Provides pre-made environments for training
- Bridges game development and machine learning
- Developed by Unity Technologies

**Learning Environment**
- The Unity scene containing environment
- Includes game characters, objects, and physics
- Built using C# and Unity Engine
- Provides simulated world for agents

**Academy**
- Component orchestrating agents and decisions
- Teacher handling Python API requests
- Synchronizes all agents
- Manages episode lifecycle
- Ensures agents stay in sync

**Agent**
- Actor in the scene being trained
- Receives observations, takes actions, gets rewards
- Policy (Brain) determines action selection
- Optimized during training process

**Policy (Brain)**
- Function mapping states to actions
- $\pi(a|s)$ = probability of action $a$ in state $s$
- Can be deterministic or stochastic
- Learned through training algorithms

### Environment Components

**Raycasts**
- Laser-like sensors detecting objects
- Provide distance and object type information
- More efficient than camera vision
- Low-dimensional structured input
- Similar to sonar or lidar

**Discrete Action Space**
- Finite set of possible actions
- Agent selects one action per timestep
- Examples: Move Forward, Rotate Left, Shoot
- Represented as integers 0, 1, 2, ...
- Used in both SnowballTarget and Pyramids

**Continuous Action Space**
- Actions represented as real numbers
- Values typically in range [-1, 1]
- Examples: Steering angle, force magnitude
- More challenging to learn than discrete
- Not used in Unit 5 environments

### Reward Concepts

**Extrinsic Reward**
- Provided by environment
- Designed by humans
- Task-specific
- Can be sparse or dense
- $R_{extrinsic}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$

**Intrinsic Reward (Curiosity)**
- Generated by agent itself
- Based on novelty or surprise
- Task-agnostic (general purpose)
- Encourages exploration
- $R_{intrinsic}: \mathcal{S} \times \mathcal{A} \times \mathcal{S}' \rightarrow \mathbb{R}$

**Sparse Rewards**
- Most rewards are zero
- Informative rewards rare
- Makes learning difficult
- Requires many episodes to learn
- Solved by curiosity or reward shaping

**Reward Engineering Problem**
- Creating overly complex reward functions
- Trying to force specific behaviors
- Can miss creative agent strategies
- Makes debugging difficult
- Best practice: Start simple

### Curiosity Methods

**Intrinsic Curiosity Module (ICM)**
- Feature-based curiosity method
- Uses three networks: feature, forward, inverse
- Predicts next state in feature space
- Inverse model ensures meaningful features
- More complex than RND

**Random Network Distillation (RND)**
- Simpler curiosity method
- Two networks: random target, trained predictor
- Measures how well agent predicts random network
- Used by ML-Agents
- Often more effective than ICM

**Forward Model**
- Predicts next state from current state and action
- $\hat{s}_{t+1} = f_\theta(s_t, a_t)$
- Prediction error measures curiosity
- Trained simultaneously with policy
- Core component of ICM

**Inverse Model**
- Predicts action from state transition
- $\hat{a}_t = h_\omega(s_t, s_{t+1})$
- Self-supervised learning component
- Ensures features encode action-relevant information
- Part of ICM architecture

### Training Algorithms

**PPO (Proximal Policy Optimization)**
- Policy gradient algorithm
- Primary algorithm in ML-Agents
- Stable and sample-efficient
- Uses clipped surrogate objective
- Works well with curiosity

**SAC (Soft Actor-Critic)**
- Off-policy algorithm
- Maximizes entropy for exploration
- Works with continuous actions
- Alternative to PPO in ML-Agents
- More sample-efficient than PPO

**Policy Gradient**
- Class of algorithms optimizing policy directly
- Uses gradient ascent on expected return
- Includes: REINFORCE, A2C, PPO
- Contrasts with value-based methods (Q-Learning)
- Better for continuous action spaces

### ML-Agents Specific

**External Communicator**
- Bridge between Unity (C#) and Python
- Enables bidirectional communication
- Passes observations and actions
- Handles environment configuration
- Critical for ML-Agents functionality

**Python Low-Level API**
- Interface for environment interaction
- Controls simulation stepping
- Launches training
- Written in Python
- Used by training scripts

**Gym Wrapper**
- Encapsulates environment in OpenAI Gym format
- Provides standard interface
- Compatible with Gym-based tools
- For single-agent environments
- Familiar API for researchers

**PettingZoo Wrapper**
- Multi-agent version of Gym wrapper
- Handles multiple simultaneous agents
- Supports competitive/cooperative scenarios
- Standard multi-agent interface
- Used for AI vs AI environments

### Mathematical Notation

**Symbols Used**:
- $s_t$, $S_t$ = State at time $t$
- $a_t$, $A_t$ = Action at time $t$
- $r_t$, $R_t$ = Reward at time $t$
- $\pi$ = Policy
- $\theta$ = Policy parameters
- $\gamma$ = Discount factor
- $\beta$ = Intrinsic reward coefficient
- $\phi$ = Feature encoder
- $\eta$ = Curiosity scaling factor

**Key Equations**:

**Expected Return**:
$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

**Total Reward with Curiosity**:
$R_{total} = R_{extrinsic} + \beta \cdot R_{intrinsic}$

**ICM Curiosity Reward**:
$r_{curiosity} = \frac{\eta}{2} \|g_\psi(\phi(s_t), a_t) - \phi(s_{t+1})\|^2$

**RND Curiosity Reward**:
$r_{curiosity} = \|\hat{f}_{predictor}(s) - f_{target}(s)\|^2$

**Policy Objective**:
$J(\theta) = E_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$

---

## Additional Concepts

### Multi-Agent Training

**Challenges**:
- Non-stationary environment (other agents learning)
- Coordination vs competition
- Credit assignment problem
- Emergent behaviors

**Applications in ML-Agents**:
- Cooperative tasks (teammates)
- Competitive tasks (opponents)
- Mixed scenarios
- Self-play

### Self-Play

**Concept**: Agent trains against copies of itself

**Benefits**:
- Automatic curriculum learning
- Continually challenging opponents
- Emergent complexity
- No human opponents needed

**Used in**: AI vs AI competitions in course

### Curriculum Learning

**Definition**: Gradually increasing task difficulty

**Implementation**:
- Start with simple environments
- Increase complexity as agent improves
- Automatic or manual progression
- Accelerates learning

**Example**:
```
Level 1: Close targets, slow movement
   ↓
Level 2: Medium distance, normal speed
   ↓
Level 3: Far targets, fast movement
```

### Transfer Learning

**Concept**: Use knowledge from one task for another

**Methods**:
- Pre-train on simple task
- Fine-tune on target task
- Share network weights
- Domain randomization

**Benefits**:
- Faster learning
- Less data needed
- Better generalization

### Hyperparameter Tuning

**Key Hyperparameters**:

**Learning Rate**: Controls update step size
- Too high: Unstable training
- Too low: Slow learning
- Typical: 1e-4 to 3e-4

**Batch Size**: Number of samples per update
- Larger: More stable, slower
- Smaller: Faster, noisier
- Typical: 64 to 512

**Buffer Size**: Replay buffer capacity
- Larger: More diverse samples
- Smaller: More on-policy
- Typical: 2048 to 10240

**Gamma (γ)**: Discount factor
- High (0.99): Values future rewards
- Low (0.9): More myopic
- Typical: 0.95 to 0.99

**Beta (β)**: Intrinsic reward weight
- Sparse rewards: 0.01 to 0.05
- Dense rewards: 0.001 to 0.01
- Environment-dependent

### Evaluation Metrics

**Training Metrics**:
- Cumulative reward per episode
- Episode length
- Policy loss
- Value loss
- Learning rate

**Performance Metrics**:
- Success rate
- Average reward
- Time to goal
- Actions per episode

**Curiosity Metrics**:
- Intrinsic reward magnitude
- States visited
- Exploration coverage
- Prediction error

### Best Practices

**Environment Design**:
1. Start with simple rewards
2. Avoid reward engineering
3. Test with random agent first
4. Ensure observations include necessary info
5. Balance action space complexity

**Training**:
1. Monitor learning curves
2. Use TensorBoard for visualization
3. Save checkpoints regularly
4. Try different random seeds
5. Compare against baselines

**Debugging**:
1. Check reward scaling
2. Verify observation normalization
3. Test action space
4. Examine episode trajectories
5. Use smaller networks initially

---

## Summary

### Key Takeaways

1. **Unity ML-Agents**
   - Bridges game development and ML
   - Provides rich 3D environments
   - Uses industry-standard game engine
   - Enables complex agent training

2. **Architecture**
   - Six main components
   - C# and Python integration
   - Academy orchestrates agents
   - Supports single and multi-agent

3. **SnowballTarget**
   - Simple discrete actions
   - Raycasts for observations
   - Frequent rewards (dense)
   - Good for learning basics

4. **Pyramids**
   - Multi-step sequential task
   - Sparse extrinsic rewards
   - Requires curiosity
   - More challenging environment

5. **Curiosity**
   - Solves sparse reward problem
   - Intrinsic motivation
   - Encourages exploration
   - Two main methods: ICM and RND

6. **Mathematical Foundations**
   - Reward hypothesis
   - Expected cumulative return
   - Feature learning
   - Prediction error

### Progression Path

```
Basic RL Concepts
      ↓
Unity ML-Agents Setup
      ↓
SnowballTarget (Dense Rewards)
      ↓
Pyramids (Sparse Rewards + Curiosity)
      ↓
Multi-Agent Environments
      ↓
AI vs AI Competition
```

### Next Steps

**After Unit 5**:
1. Train agents in both environments
2. Experiment with hyperparameters
3. Visualize results on Hugging Face Hub
4. Prepare for multi-agent challenges
5. Learn advanced curiosity techniques

**Future Units**:
- Actor-Critic methods
- Multi-agent scenarios
- Competitive AI vs AI
- Advanced PPO techniques

---

## References and Resources

### Official Documentation
- [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [Unity Learn](https://learn.unity.com/)

### Academic Papers

**Curiosity - ICM**:
- Pathak et al. (2017), "Curiosity-driven Exploration by Self-supervised Prediction"
- arXiv:1705.05363

**Curiosity - RND**:
- Burda et al. (2018), "Exploration by Random Network Distillation"
- arXiv:1810.12894

**PPO**:
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
- arXiv:1707.06347

### Additional Reading

**Blog Posts**:
- [Curiosity-Driven Learning through Next State Prediction](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-next-state-prediction-f7f4e2f592fa)
- [Random Network Distillation: a new take on Curiosity-Driven Learning](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)

**Books**:
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
- Graesser & Keng, "Foundations of Deep Reinforcement Learning" (2020)

### Tools and Libraries

- **ML-Agents**: `pip install mlagents`
- **Unity Hub**: For Unity installation
- **TensorBoard**: Training visualization
- **Hugging Face Hub**: Model sharing

---

*These notes cover Unit 5: Introduction to Unity ML-Agents from the Hugging Face Deep RL Course. Practice training agents in both SnowballTarget and Pyramids environments to solidify your understanding of Unity ML-Agents and curiosity-driven learning!*