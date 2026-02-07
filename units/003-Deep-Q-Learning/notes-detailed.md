# Unit 3: Deep Q-Learning with Atari Games

# Table of Contents

- [1. Introduction](#1-introduction)
  - [Overview](#overview)
  - [The Scalability Problem](#the-scalability-problem)
  - [The Solution: Deep Q-Learning](#the-solution-deep-q-learning)
  - [What We'll Learn](#what-well-learn)

- [2. From Q-Learning to Deep Q-Learning](#2-from-q-learning-to-deep-q-learning)
  - [Review: Q-Learning Basics](#review-q-learning-basics)
  - [The Fundamental Problem](#the-fundamental-problem)
  - [Why Tabular Methods Fail for Atari](#why-tabular-methods-fail-for-atari)
  - [The Deep Q-Learning Solution](#the-deep-q-learning-solution)
  - [Comparison: Q-Learning vs Deep Q-Learning](#comparison-q-learning-vs-deep-q-learning)
  - [Why Neural Networks?](#why-neural-networks)

- [3. The Deep Q-Network (DQN)](#3-the-deep-q-network-dqn)
  - [DQN Architecture Overview](#dqn-architecture-overview)

  - [Input Preprocessing](#input-preprocessing)
    - [Why Preprocess?](#why-preprocess)

  - [Frame Stacking: Solving Temporal Limitation](#frame-stacking-solving-temporal-limitation)
    - [The Temporal Limitation Problem](#the-temporal-limitation-problem)
    - [Frame Stacking Solution](#frame-stacking-solution)

  - [Convolutional Layers](#convolutional-layers)
    - [Purpose](#purpose)
    - [Typical DQN Convolutional Architecture](#typical-dqn-convolutional-architecture)

  - [Fully Connected Layers](#fully-connected-layers)
    - [Purpose](#purpose-1)
    - [Architecture](#architecture)

  - [Action Selection](#action-selection)
  - [Complete DQN Architecture Summary](#complete-dqn-architecture-summary)
  - [Training Process](#training-process)

- [4. The Deep Q-Learning Algorithm](#4-the-deep-q-learning-algorithm)
  - [Overview](#overview-1)
  - [Key Difference: Loss Function Instead of Direct Update](#key-difference-loss-function-instead-of-direct-update)
  - [The Loss Function](#the-loss-function)
  - [Two-Phase Training Process](#two-phase-training-process)
    - [Phase 1: Sampling](#phase-1-sampling)
    - [Phase 2: Training](#phase-2-training)
  - [Training Instability Problem](#training-instability-problem)
  - [Three Solutions to Stabilize Training](#three-solutions-to-stabilize-training)

- [Experience Replay](#experience-replay)
  - [The Problem Without Experience Replay](#the-problem-without-experience-replay)
  - [What is Experience Replay?](#what-is-experience-replay)
  - [How Experience Replay Works](#how-experience-replay-works)
  - [Benefits of Experience Replay](#benefits-of-experience-replay)
  - [Experience Replay in Pseudocode](#experience-replay-in-pseudocode)
  - [Implementation Considerations](#implementation-considerations)

- [Fixed Q-Target](#fixed-q-target)
  - [The Moving Target Problem](#the-moving-target-problem)
  - [The Cowboy and Cow Analogy](#the-cowboy-and-cow-analogy)
  - [The Fixed Q-Target Solution](#the-fixed-q-target-solution)
  - [Implementation](#implementation)
  - [Choosing Update Frequency C](#choosing-update-frequency-c)

- [Double DQN](#double-dqn)
  - [The Q-Value Overestimation Problem](#the-q-value-overestimation-problem)
  - [The Double DQN Solution](#the-double-dqn-solution)
  - [Implementation](#implementation-1)
  - [Benefits of Double DQN](#benefits-of-double-dqn)
  - [When to Use Double DQN](#when-to-use-double-dqn)

- [Complete Deep Q-Learning Algorithm](#complete-deep-q-learning-algorithm)
  - [Algorithm Pseudocode](#algorithm-pseudocode)
  - [Key Hyperparameters](#key-hyperparameters)
  - [Training Tips](#training-tips)

- [5. Glossary](#5-glossary)

## Appendix

- [Comparison: Q-Learning vs Deep Q-Learning](#comparison-q-learning-vs-deep-q-learning)
- [Practical Implementation Guide](#practical-implementation-guide)
- [Performance Benchmarks](#performance-benchmarks)
- [Advanced Topics](#advanced-topics)
- [Summary](#summary)
- [References and Resources](#references-and-resources)


---

## 1. Introduction

### Overview

In Unit 2, we learned Q-Learning and implemented it from scratch in simple environments:
- **FrozenLake-v1**: 16 states
- **Taxi-v3**: 500 states

These environments had **discrete and small state spaces**, making Q-tables practical and effective.

### The Scalability Problem

**The Challenge**: Q-Learning is a **tabular method** that becomes ineffective with large state spaces.

**Example - Atari Games**:
- State space in Atari can contain **10^9 to 10^11 states**
- Space Invaders frame: (210, 160, 3) pixels
  - 210 × 160 pixels
  - 3 color channels (RGB)
  - Values range from 0 to 255
- **Possible observations**: 256^(210×160×3) = 256^100,800

**For Comparison**: The observable universe has approximately 10^80 atoms!

### The Solution: Deep Q-Learning

Instead of using a Q-table, Deep Q-Learning uses a **Neural Network** that:
- Takes a state as input
- Approximates Q-values for each action based on that state
- Scales to complex environments with massive state spaces

### What We'll Learn

In this unit, we'll:
- Understand why Deep Q-Learning is necessary
- Learn the architecture of Deep Q-Networks (DQN)
- Study the Deep Q-Learning algorithm
- Train agents to play Atari games (Space Invaders, etc.)
- Use RL-Zoo framework for training and evaluation

**Tools**: 
- [RL-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo): Training framework for RL using Stable-Baselines3
- Provides scripts for training, evaluation, hyperparameter tuning, plotting, and video recording

---

## 2. From Q-Learning to Deep Q-Learning

### Review: Q-Learning Basics

**Q-Learning** is an algorithm that trains a Q-Function:
- **Q-Function**: Action-value function
- **Determines**: Value of being at a particular state and taking a specific action
- **Q stands for**: "Quality" of that action at that state

**Implementation**: Internally encoded by a **Q-table**
- Table where each cell = state-action pair value
- Think of it as the "memory" or "cheat sheet" of the Q-function

### The Fundamental Problem

**Q-Learning is a tabular method**, which means:
- It's **not scalable** to large state/action spaces
- Cannot be represented efficiently by arrays and tables
- Memory requirements grow exponentially

### Why Tabular Methods Fail for Atari

**Atari Environment Complexity** (as noted by Nikita Melkozerov):

**Single Frame Composition**:
```
Dimensions: 210 × 160 pixels
Channels: 3 (RGB color)
Shape: (210, 160, 3)
Value range per pixel: 0 to 255
```

**State Space Calculation**:
```
Total possible observations = 256^(210 × 160 × 3)
                            = 256^100,800
                            ≈ Incomprehensibly large!
```

**Q-Table Requirements**:
- Would need to store Q-values for each state-action pair
- Memory needed: State space × Action space
- **Completely impractical** for Atari-scale environments

### The Deep Q-Learning Solution

Instead of a Q-table, use a **parametrized Q-function**: Q_θ(s, a)

**Key Idea**:
- Use a **neural network** to approximate Q-values
- Network learns parameters θ (weights)
- Given a state, network outputs Q-values for all possible actions

**How It Works**:
```
Input: State (e.g., game frame)
       ↓
   Neural Network (parameters θ)
       ↓
Output: Vector of Q-values [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

### Comparison: Q-Learning vs Deep Q-Learning

| Aspect | Q-Learning | Deep Q-Learning |
|--------|------------|----------------|
| **Value Storage** | Q-table (discrete) | Neural network (continuous) |
| **State Space** | Small, discrete | Large, continuous possible |
| **Scalability** | Limited | Highly scalable |
| **Memory** | O(states × actions) | O(network parameters) |
| **Generalization** | None (exact lookup) | Yes (learns patterns) |
| **Update Method** | Direct table update | Gradient descent |
| **Best For** | Simple environments | Complex environments |

### Why Neural Networks?

**Advantages**:
1. **Generalization**: Learn patterns across similar states
2. **Continuous States**: Handle high-dimensional input
3. **Function Approximation**: Approximate complex Q-functions
4. **Compact Representation**: Fixed network size regardless of state space
5. **Feature Learning**: Automatically learn relevant features

**Example**:
- Two similar game frames (ball moved slightly) should have similar Q-values
- Neural network can learn this similarity
- Q-table treats them as completely separate states

---

## 3. The Deep Q-Network (DQN)

### DQN Architecture Overview

The Deep Q-Network architecture consists of:
1. **Input Processing**: Stack of 4 preprocessed frames
2. **Convolutional Layers**: Extract spatial features
3. **Fully Connected Layers**: Combine features and output Q-values
4. **Output**: Vector of Q-values for each possible action

```
Input: Stack of 4 frames (84 × 84 × 4)
       ↓
Convolutional Layer 1
       ↓
Convolutional Layer 2
       ↓
Convolutional Layer 3
       ↓
Fully Connected Layer 1
       ↓
Fully Connected Layer 2
       ↓
Output: Q-values for each action [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

### Input Preprocessing

#### Why Preprocess?

**Goal**: Reduce complexity of state to decrease computation time

**Key Preprocessing Steps**:

1. **Grayscale Conversion**
   - Original: 3 color channels (RGB)
   - Processed: 1 grayscale channel
   - **Benefit**: Reduces channels by 66%
   - **Rationale**: Colors don't add critical information in most Atari games

2. **Resize to 84×84**
   - Original: 210 × 160 pixels
   - Processed: 84 × 84 pixels
   - **Benefit**: Reduces pixel count by ~85%
   - **Rationale**: Smaller size sufficient for game information

3. **Cropping** (game-specific)
   - Remove parts of screen without important information
   - Example: Score displays, borders, static elements
   - **Benefit**: Further complexity reduction

**Size Comparison**:
```
Original frame: 210 × 160 × 3 = 100,800 values
Processed frame: 84 × 84 × 1 = 7,056 values
Reduction: ~93% fewer values per frame!
```

### Frame Stacking: Solving Temporal Limitation

#### The Temporal Limitation Problem

**Problem**: A single frame doesn't contain temporal information (motion)

**Example - Pong Game**:
```
Single frame: 
[Image of ball and paddle]
Question: Where is the ball going?
Answer: Impossible to tell! No motion information.
```

```
Four stacked frames:
[Ball position at t-3]
[Ball position at t-2]
[Ball position at t-1]
[Ball position at t]
Answer: Ball is moving to the right!
```

#### Frame Stacking Solution

**Approach**: Stack 4 consecutive frames together

**Benefits**:
1. **Capture Motion**: See trajectory of objects
2. **Temporal Context**: Understand direction and speed
3. **Better Decisions**: Agent can predict future positions

**Implementation**:
```
Input shape: (84, 84, 4)
- 84 × 84 pixels
- 4 frames stacked (temporal dimension)
```

**Why 4 frames?**
- Empirically found to work well
- Balance between temporal info and computational cost
- Enough to capture motion but not too much memory

### Convolutional Layers

#### Purpose

**Convolutional layers** allow the network to:
1. **Capture spatial relationships** in images
2. **Exploit local patterns** (edges, shapes, objects)
3. **Learn hierarchical features** (low-level → high-level)
4. **Share parameters** across spatial locations (efficiency)

**Temporal Exploitation**:
- Because frames are stacked, convolutions can also capture temporal properties
- Learn patterns of motion and change across frames

#### Typical DQN Convolutional Architecture

**Layer 1**:
```
Input: 84 × 84 × 4
Filters: 32
Kernel size: 8 × 8
Stride: 4
Activation: ReLU
Output: 20 × 20 × 32
```

**Layer 2**:
```
Input: 20 × 20 × 32
Filters: 64
Kernel size: 4 × 4
Stride: 2
Activation: ReLU
Output: 9 × 9 × 64
```

**Layer 3**:
```
Input: 9 × 9 × 64
Filters: 64
Kernel size: 3 × 3
Stride: 1
Activation: ReLU
Output: 7 × 7 × 64
```

**Feature Progression**:
- **Layer 1**: Detects edges, basic shapes
- **Layer 2**: Detects object parts, textures
- **Layer 3**: Detects whole objects, complex patterns

### Fully Connected Layers

#### Purpose

After convolutional layers extract features, fully connected layers:
1. **Combine all features** from across the image
2. **Learn action values** based on complete state
3. **Output Q-values** for decision making

#### Architecture

**Flatten Layer**:
```
Input: 7 × 7 × 64 = 3,136 features
Output: 3,136-dimensional vector
```

**Fully Connected Layer 1**:
```
Input: 3,136 features
Neurons: 512
Activation: ReLU
Output: 512 features
```

**Fully Connected Layer 2 (Output)**:
```
Input: 512 features
Neurons: Number of actions (e.g., 4 for Pong)
Activation: None (linear)
Output: Q-value for each action
```

### Action Selection

After network outputs Q-values, use **epsilon-greedy policy**:

```python
if random() < epsilon:
    action = random_action()  # Exploration
else:
    action = argmax(Q_values)  # Exploitation
```

**Example Output**:
```
Q-values: [2.3, 5.1, 1.8, 4.2]
Actions:  [Left, Right, Up, Down]

With ε = 0.1:
- 90% of time: Choose Right (highest Q-value = 5.1)
- 10% of time: Choose random action
```

### Complete DQN Architecture Summary

```
Input: 84 × 84 × 4 (4 stacked frames, grayscale, resized)
   ↓
Conv Layer 1: 32 filters, 8×8, stride 4, ReLU → 20×20×32
   ↓
Conv Layer 2: 64 filters, 4×4, stride 2, ReLU → 9×9×64
   ↓
Conv Layer 3: 64 filters, 3×3, stride 1, ReLU → 7×7×64
   ↓
Flatten: → 3,136 features
   ↓
FC Layer 1: 512 neurons, ReLU → 512 features
   ↓
FC Layer 2: n_actions neurons, Linear → Q-values
   ↓
Output: [Q(s,a₁), Q(s,a₂), ..., Q(s,aₙ)]
```

### Training Process

**Initial State**:
- Network weights are randomly initialized
- Q-value estimates are terrible
- Agent plays poorly

**During Training**:
- Agent explores environment
- Network learns to associate situations with appropriate actions
- Q-value estimates improve
- Performance increases

**Final State**:
- Network has learned good Q-function approximation
- Agent can play game well
- Near-optimal policy achieved

---

## 4. The Deep Q-Learning Algorithm

### Overview

Deep Q-Learning uses a neural network to approximate Q-values, but the training process differs from standard Q-Learning.

### Key Difference: Loss Function Instead of Direct Update

**Q-Learning (Tabular)**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```
Direct update to Q-table entry

**Deep Q-Learning**:
```
Loss = (Q_target - Q_prediction)²
Update network weights using gradient descent
```
Minimize loss to approximate Q-values better

### The Loss Function

**TD Target (Q-Target)**:
```
Q_target = r + γ max_a' Q_θ(s', a')
```

**Current Q-Value (Prediction)**:
```
Q_prediction = Q_θ(s, a)
```

**Loss Function** (Mean Squared Error):
```
Loss = (Q_target - Q_prediction)²
     = (r + γ max_a' Q_θ(s', a') - Q_θ(s, a))²
```

**Gradient Descent Update**:
```
θ ← θ - α ∇_θ Loss
```
Update network parameters to minimize loss

### Two-Phase Training Process

#### Phase 1: Sampling

**Purpose**: Collect experience from environment

**Process**:
1. Agent performs actions in environment
2. Observes outcomes (state, action, reward, next_state)
3. Stores experience tuples in **replay memory**

**Experience Tuple**:
```
(s_t, a_t, r_{t+1}, s_{t+1}, done)
```
- s_t: Current state
- a_t: Action taken
- r_{t+1}: Reward received
- s_{t+1}: Next state
- done: Whether episode terminated

#### Phase 2: Training

**Purpose**: Learn from collected experiences

**Process**:
1. Sample small **batch of tuples randomly** from replay memory
2. Compute Q-targets for batch
3. Compute Q-predictions for batch
4. Calculate loss
5. Update network weights using **gradient descent**

**Batch Learning**:
```python
# Sample random batch from replay memory
batch = replay_memory.sample(batch_size=32)

# Compute loss for batch
loss = mean((Q_target - Q_prediction)²)

# Update network
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Training Instability Problem

**Why Deep Q-Learning Can Be Unstable**:

1. **Non-linear Function Approximation**
   - Neural networks are non-linear
   - Small parameter changes can cause large output changes
   - Can lead to divergence

2. **Bootstrapping**
   - Update targets using existing estimates (not actual returns)
   - Errors can accumulate and amplify
   - Can cause oscillations

**Result**: Combining neural networks + bootstrapping can cause training instability

### Three Solutions to Stabilize Training

To address instability, Deep Q-Learning implements three key improvements:

1. **Experience Replay**: Make more efficient use of experiences
2. **Fixed Q-Target**: Stabilize the training process
3. **Double DQN**: Handle overestimation of Q-values

Let's examine each solution in detail.

---

## Experience Replay

### The Problem Without Experience Replay

**Typical Online RL**:
```
1. Agent interacts with environment
2. Gets experience (s, a, r, s')
3. Learns from it immediately (updates network)
4. Discards the experience
5. Gets new experience
6. Repeat
```

**Issues**:
1. **Inefficient**: Each experience used only once
2. **Correlation**: Sequential experiences are highly correlated
3. **Catastrophic Forgetting**: New experiences overwrite old knowledge

### What is Experience Replay?

**Core Idea**: Store experiences in a memory buffer and reuse them

**Replay Memory (Replay Buffer)**:
- Data structure storing experience tuples
- Fixed capacity N (e.g., 1,000,000 experiences)
- Oldest experiences removed when buffer is full (FIFO)

**Structure**:
```python
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, experience):
        # Add experience: (s, a, r, s', done)
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)  # Remove oldest
        self.memory.append(experience)
        
    def sample(self, batch_size):
        # Return random batch of experiences
        return random.sample(self.memory, batch_size)
```

### How Experience Replay Works

**During Episode**:
1. Agent takes action, observes outcome
2. Store experience tuple in replay memory
3. Continue interaction

**During Training**:
1. Sample random batch from replay memory
2. Learn from this batch
3. Update network weights

**Key Point**: Same experiences can be sampled multiple times!

### Benefits of Experience Replay

#### 1. More Efficient Use of Experiences

**Before**:
- Experience used once, then discarded
- Wasteful, especially in sample-inefficient environments

**After**:
- Experience stored in memory
- Can be used multiple times during training
- Learn more from same amount of interaction

**Example**:
```
Single experience can be:
- Used in batch at step 1000
- Used in batch at step 2500
- Used in batch at step 4200
- etc.
```

#### 2. Break Correlation Between Sequential Experiences

**Problem with Sequential Learning**:
```
Time 0: State=Level1_Position1, Action=Right, Reward=+1
Time 1: State=Level1_Position2, Action=Right, Reward=+1
Time 2: State=Level1_Position3, Action=Right, Reward=+1
```
- All experiences highly correlated
- Network learns biased patterns
- Can cause action values to oscillate or diverge

**Solution with Random Sampling**:
```
Batch might contain:
- Experience from Level 1, Position 1
- Experience from Level 3, Position 7
- Experience from Level 2, Position 4
- Experience from Level 1, Position 9
```
- Experiences uncorrelated
- More diverse training signal
- More stable learning

#### 3. Avoid Catastrophic Forgetting

**Catastrophic Forgetting (Catastrophic Interference)**:
- Neural network forgets previous experiences as it learns new ones
- Especially problematic with sequential similar experiences

**Example**:
```
Agent trains on Level 1:
- Learns how to navigate Level 1 well

Agent moves to Level 2 (different layout):
- Network weights updated for Level 2
- Forgets how to play Level 1!

Agent returns to Level 1:
- Performance has degraded significantly
```

**How Experience Replay Helps**:
- Replay buffer contains experiences from all levels
- Random sampling ensures mixed training
- Network maintains knowledge of all experiences
- Prevents forgetting previous situations

### Experience Replay in Pseudocode

```
Initialize replay memory D with capacity N
Initialize Q-network with random weights θ

For episode = 1 to M:
    Initialize state s
    
    For t = 1 to T:
        # Sample phase
        Select action a using ε-greedy policy
        Execute action a
        Observe reward r and next state s'
        Store transition (s, a, r, s', done) in D
        
        # Training phase (every step or every k steps)
        Sample random minibatch of transitions from D
        For each transition in minibatch:
            Compute Q_target = r + γ max_a' Q(s', a')
            Compute loss = (Q_target - Q(s, a))²
        
        Update network weights θ using gradient descent
        
        s ← s'
```

### Implementation Considerations

**Buffer Size N**:
- Typical: 100,000 to 1,000,000 experiences
- Larger = more diverse experiences, more memory
- Smaller = less memory, but less diversity

**Batch Size**:
- Typical: 32 to 128 experiences per batch
- Larger = more stable gradients, slower updates
- Smaller = faster updates, noisier gradients

**When to Train**:
- Every step: Maximum learning
- Every k steps: More efficient
- Trade-off between learning speed and computation

---

## Fixed Q-Target

### The Moving Target Problem

#### Understanding TD Target Calculation

**TD Target Formula**:
```
Q_target = r + γ max_a' Q_θ(s', a')
```

**Loss Calculation**:
```
Loss = (Q_target - Q_θ(s, a))²
```

**The Problem**:
- We use the **same network parameters θ** to compute both:
  1. Q_target (what we want Q-values to be)
  2. Q_θ(s, a) (our current Q-value estimate)

#### Why This Causes Instability

**At every training step**:
1. We update parameters θ to match Q_target
2. But Q_target also changes because it uses parameters θ!
3. We're chasing a moving target

**Result**:
- Both Q-value and target shift at each step
- Getting closer to target, but target also moving
- Can cause significant oscillation in training

### The Cowboy and Cow Analogy

**Setup**:
- You're a cowboy (Q-value estimation)
- Goal: Catch the cow (Q-target)
- Your objective: Get closer (reduce error)

**What Happens**:

**Step 1**:
```
Cowboy position: ●
Cow position:         ○
You move closer →
```

**Step 2**:
```
Cowboy position:   ●
Cow position:           ○  (cow also moved!)
You move closer →
```

**Step 3**:
```
Cowboy position:     ●
Cow position:              ○  (cow moved again!)
You move closer →
```

**Result**: Bizarre chase pattern, significant oscillation, unstable training!

### The Fixed Q-Target Solution

#### Core Idea

Instead of using the same network for both Q-value and Q-target:

1. **Use separate network** with **fixed parameters** for Q-target
2. **Copy parameters** from main network every C steps

**Two Networks**:
- **Q-Network (θ)**: Main network, updated every step
- **Target Network (θ⁻)**: Separate network, updated every C steps

#### How It Works

**Target Calculation**:
```
Q_target = r + γ max_a' Q_θ⁻(s', a')  ← Uses target network!
```

**Current Q-Value**:
```
Q_prediction = Q_θ(s, a)  ← Uses main network
```

**Loss**:
```
Loss = (Q_target - Q_prediction)²
```

**Network Updates**:
```
Every step:
  - Update Q-network parameters: θ ← θ - α ∇Loss

Every C steps:
  - Copy Q-network to target network: θ⁻ ← θ
```

#### Why This Stabilizes Training

**Before (Moving Target)**:
```
Step 1: Q_target = r + γ max Q_θ(s', a')
        Update θ
        
Step 2: Q_target = r + γ max Q_θ(s', a')  ← Changed!
        Update θ
        
Step 3: Q_target = r + γ max Q_θ(s', a')  ← Changed again!
```
Target constantly moving

**After (Fixed Target)**:
```
Step 1-C: Q_target = r + γ max Q_θ⁻(s', a')  ← Fixed!
          Update only θ (not θ⁻)
          
Step C: Copy θ to θ⁻
        
Step C+1 to 2C: Q_target = r + γ max Q_θ⁻(s', a')  ← Fixed again!
```
Target stays fixed for C steps

**Benefits**:
1. **Stable target** for Q-network to learn
2. **Reduced oscillation** in training
3. **More consistent** learning signal
4. **Better convergence** properties

### Implementation

```python
class DQN:
    def __init__(self, ...):
        # Main Q-network
        self.q_network = QNetwork(...)
        
        # Target network (separate copy)
        self.target_network = QNetwork(...)
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )
        
        self.update_counter = 0
        self.C = 1000  # Update target every C steps
        
    def train_step(self, batch):
        # Compute Q-predictions using Q-network
        q_pred = self.q_network(states)[actions]
        
        # Compute Q-targets using TARGET network
        with torch.no_grad():  # Don't compute gradients
            q_next = self.target_network(next_states).max(1)[0]
            q_target = rewards + gamma * q_next
        
        # Compute loss
        loss = (q_target - q_pred).pow(2).mean()
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network every C steps
        self.update_counter += 1
        if self.update_counter % self.C == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )
```

### Choosing Update Frequency C

**Typical Values**: C = 1,000 to 10,000 steps

**Trade-offs**:

**Small C** (e.g., 100):
- Target updates frequently
- Faster adaptation to Q-network changes
- Still some instability possible
- Closer to moving target problem

**Large C** (e.g., 10,000):
- Target stays fixed longer
- More stable training
- Slower adaptation to Q-network improvements
- May use outdated targets

**Common Choice**: C = 1,000 to 5,000 (good balance)

---

## Double DQN

### The Q-Value Overestimation Problem

#### How We Calculate TD Target

**Standard DQN TD Target**:
```
Q_target = r + γ max_a' Q_θ⁻(s', a')
```

**The max operation**: Select action with highest Q-value

#### Why This Causes Overestimation

**The Question**: How do we know the action with highest Q-value is actually the best action?

**The Issue**:
- Q-values depend on what actions we've tried
- Q-values depend on what states we've explored
- At beginning of training: **limited knowledge**
- Q-values are **noisy** (not accurate yet)

**Result**:
- Taking max of noisy values tends to select overestimated values
- Non-optimal actions may temporarily have higher Q-values
- Leads to **positive bias** (systematic overestimation)
- Makes learning difficult

#### Example of Overestimation

```
True Q-values:
Q(s', left) = 5.0
Q(s', right) = 7.0  ← Truly best action

Noisy Estimates:
Q(s', left) = 8.0  ← Overestimated!
Q(s', right) = 6.0 ← Underestimated!

max_a Q(s', a) = 8.0  ← Selected wrong action!
```

**If this happens frequently**:
- Agent learns incorrect Q-values
- Training becomes unstable
- Convergence slows or fails

### The Double DQN Solution

#### Core Idea

**Decouple action selection from Q-value estimation** using two networks:

1. **DQN Network**: Select best action
2. **Target Network**: Evaluate that action

#### How Double DQN Works

**Step 1: Action Selection** (use DQN network)
```
a* = argmax_a' Q_θ(s', a')
```
DQN network chooses which action looks best

**Step 2: Action Evaluation** (use Target network)
```
Q_target = r + γ Q_θ⁻(s', a*)
```
Target network evaluates the chosen action

**Complete Formula**:
```
Q_target = r + γ Q_θ⁻(s', argmax_a' Q_θ(s', a'))
```

#### Why This Reduces Overestimation

**Standard DQN (Single Network)**:
```
# Same network for both selection and evaluation
a* = argmax_a Q(s', a)  ← Network 1
Q_target = r + γ Q(s', a*)  ← Same network 1
```
If Q(s', a*) is overestimated, bias propagates

**Double DQN (Two Networks)**:
```
# Different networks for selection and evaluation
a* = argmax_a Q_θ(s', a)  ← DQN network
Q_target = r + γ Q_θ⁻(s', a*)  ← Target network
```
Even if Q_θ overestimates a*, Q_θ⁻ provides independent estimate

**Key Insight**:
- Unlikely both networks overestimate same action
- Two independent estimates reduce bias
- More accurate Q-target

#### Comparison

**Standard DQN**:
```
If DQN network thinks action A is best (even if wrong):
→ Uses that same network's Q-value for action A
→ Overestimation compounds
```

**Double DQN**:
```
If DQN network thinks action A is best:
→ Uses TARGET network's Q-value for action A
→ Independent evaluation reduces overestimation
```

### Implementation

```python
def compute_td_target_double_dqn(
    rewards, next_states, dones, gamma
):
    with torch.no_grad():
        # Step 1: Use Q-network to select best actions
        q_next_online = q_network(next_states)
        best_actions = q_next_online.argmax(dim=1)
        
        # Step 2: Use Target network to evaluate those actions
        q_next_target = target_network(next_states)
        q_next = q_next_target.gather(1, best_actions.unsqueeze(1))
        
        # Step 3: Compute TD target
        td_target = rewards + gamma * q_next * (1 - dones)
    
    return td_target
```

**Compare with Standard DQN**:
```python
def compute_td_target_standard_dqn(
    rewards, next_states, dones, gamma
):
    with torch.no_grad():
        # Single network does both selection and evaluation
        q_next = target_network(next_states).max(dim=1)[0]
        td_target = rewards + gamma * q_next * (1 - dones)
    
    return td_target
```

### Benefits of Double DQN

1. **Reduces overestimation bias**
   - More accurate Q-value estimates
   - Better learning signal

2. **Faster training**
   - More stable Q-values
   - Quicker convergence

3. **Better final performance**
   - More optimal policies learned
   - Higher scores achieved

4. **More stable learning**
   - Less oscillation in Q-values
   - Smoother training curves

### When to Use Double DQN

**Always recommended for Deep Q-Learning!**
- Minimal computational overhead
- Significant performance improvement
- Now standard in modern implementations

---

## Complete Deep Q-Learning Algorithm

### Algorithm Pseudocode

```
# Initialization
Initialize replay memory D with capacity N
Initialize Q-network with random weights θ
Initialize target network with weights θ⁻ = θ
Set update counter = 0

For episode = 1 to M:
    Initialize state s (preprocess and stack 4 frames)
    
    For t = 1 to T:
        # ACTION SELECTION
        With probability ε:
            Select random action a
        Otherwise:
            Select a = argmax_a Q_θ(s, a)
        
        # ENVIRONMENT INTERACTION
        Execute action a in environment
        Observe reward r and next state s'
        Preprocess and stack s' with previous 3 frames
        
        # STORE EXPERIENCE
        Store transition (s, a, r, s', done) in D
        
        # TRAINING (if enough samples)
        If len(D) > batch_size:
            # Sample random minibatch from D
            Sample batch of transitions (s, a, r, s', done)
            
            # Compute TD targets (Double DQN)
            For each transition:
                If done:
                    y = r
                Else:
                    # Double DQN: use Q-network to select, Target to evaluate
                    a* = argmax_a' Q_θ(s', a')
                    y = r + γ Q_θ⁻(s', a*)
            
            # Compute loss
            Loss = mean((y - Q_θ(s, a))²)
            
            # Update Q-network
            Perform gradient descent step on Loss with respect to θ
            
            # Update target network every C steps
            update_counter += 1
            If update_counter % C == 0:
                θ⁻ ← θ
        
        # Update state
        s ← s'
        
        # Decay epsilon
        ε ← max(ε_min, ε * ε_decay)
```

### Key Hyperparameters

| Hyperparameter | Typical Value | Description |
|----------------|---------------|-------------|
| **N** (replay buffer size) | 100,000 - 1,000,000 | Capacity of experience replay memory |
| **batch_size** | 32 - 128 | Number of experiences per training batch |
| **γ** (discount factor) | 0.95 - 0.99 | Future reward discount |
| **ε_start** (initial epsilon) | 1.0 | Starting exploration rate |
| **ε_min** (minimum epsilon) | 0.01 - 0.1 | Final exploration rate |
| **ε_decay** | 0.995 - 0.9995 | Epsilon decay per episode |
| **C** (target update freq) | 1,000 - 10,000 | Steps between target network updates |
| **learning_rate** | 0.00025 - 0.001 | Neural network learning rate |
| **n_frames** | 4 | Number of frames to stack |

### Training Tips

#### 1. Preprocessing
```python
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84))
    # Normalize to [0, 1]
    normalized = resized / 255.0
    return normalized
```

#### 2. Frame Stacking
```python
class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, initial_frame):
        for _ in range(self.n_frames):
            self.frames.append(initial_frame)
    
    def add(self, frame):
        self.frames.append(frame)
    
    def get_state(self):
        return np.stack(self.frames, axis=0)
```

#### 3. Epsilon Decay Strategies

**Linear Decay**:
```python
epsilon = max(epsilon_min, 
              epsilon_start - (epsilon_start - epsilon_min) * (step / total_steps))
```

**Exponential Decay**:
```python
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

**Step Decay**:
```python
if episode % 100 == 0:
    epsilon *= 0.9
```

#### 4. Reward Clipping
```python
# Clip rewards to [-1, 1] for stability
reward = np.clip(reward, -1, 1)
```

---

## 5. Glossary

### Core Concepts

**Tabular Method**
- **Definition**: RL approach where state and action spaces are small enough to be represented as arrays and tables
- **Example**: Q-Learning with Q-table
- **Limitation**: Not scalable to large state spaces
- **When to use**: Simple environments with discrete, small state spaces

**Deep Q-Learning (DQN)**
- **Definition**: Method that trains neural network to approximate Q-values for each possible action at a state
- **Purpose**: Solve problems where observational space is too big for tabular Q-Learning
- **Key innovation**: Uses function approximation instead of table lookup
- **Applications**: Atari games, robotics, complex control tasks

**Temporal Limitation**
- **Problem**: Single frame doesn't provide temporal information (motion)
- **Why it matters**: Can't determine velocity, direction, or trajectory from one frame
- **Solution**: Stack multiple frames together (typically 4)
- **Example**: In Pong, need multiple frames to see ball direction

### Training Phases

**Sampling Phase**
- **Actions**: Agent performs actions in environment
- **Observation**: Records experience tuples (s, a, r, s', done)
- **Storage**: Stores observed experiences in replay memory
- **Frequency**: Continuous during episode execution

**Training Phase**
- **Selection**: Random batches of tuples selected from replay memory
- **Learning**: Neural network weights updated using gradient descent
- **Frequency**: Every step or every k steps
- **Goal**: Minimize loss between Q-prediction and Q-target

### Stabilization Solutions

**Experience Replay**
- **Purpose**: Make more efficient use of experiences
- **Mechanism**: Replay memory saves experience samples for reuse during training
- **Benefits**:
  - Learn from same experiences multiple times
  - Avoid catastrophic forgetting
  - Break correlation in observation sequences
  - Prevent action value oscillation

**Random Sampling**
- **Purpose**: Remove correlation in observation sequences
- **How**: Select experiences randomly from replay buffer
- **Benefits**:
  - Prevents action values from oscillating
  - Avoids catastrophic divergence
  - Provides diverse training signal
  - Reduces overfitting to recent experiences

**Fixed Q-Target**
- **Problem**: Same network weights used for Q-target and Q-value calculation
- **Issue**: Every time Q-value modified, Q-target also moves
- **Solution**: Separate network with fixed parameters for TD target estimation
- **Implementation**: Target network updated by copying from DQN every C steps
- **Benefit**: Stable learning target, reduced oscillation

**Double DQN**
- **Problem**: Overestimation of Q-values
- **Cause**: Using max operator on noisy estimates
- **Solution**: Use two networks to decouple action selection from value generation
  - **DQN Network**: Selects best action for next state
  - **Target Network**: Calculates target Q-value for that action
- **Benefits**:
  - Reduces Q-value overestimation
  - Faster training
  - More stable learning
  - Better final performance

### Additional Improvements (Beyond Course Scope)

**Prioritized Experience Replay**
- **Idea**: Sample important experiences more frequently
- **Importance**: Based on TD error magnitude
- **Benefit**: Learn faster from important transitions

**Dueling DQN**
- **Architecture**: Split Q-network into two streams
  - Value stream: V(s)
  - Advantage stream: A(s,a)
- **Combination**: Q(s,a) = V(s) + A(s,a)
- **Benefit**: Better value estimation, especially for states where actions don't matter much

**Rainbow DQN**
- **Concept**: Combination of multiple DQN improvements
- **Includes**: Double DQN, Dueling DQN, Prioritized Replay, Multi-step Learning, Distributional RL, Noisy Networks
- **Result**: State-of-the-art performance

---

## Comparison: Q-Learning vs Deep Q-Learning

### Side-by-Side Comparison

| Aspect | Q-Learning | Deep Q-Learning |
|--------|------------|----------------|
| **Value Storage** | Q-table (discrete lookup) | Neural network (function approximation) |
| **State Space** | Small, discrete | Large, continuous possible |
| **Memory Complexity** | O(states × actions) | O(network parameters) |
| **Update Method** | Direct table update | Gradient descent |
| **Generalization** | None (exact values only) | Yes (learns patterns) |
| **Scalability** | Poor for large spaces | Excellent |
| **Sample Efficiency** | Moderate | Lower (but handles complex envs) |
| **Computational Cost** | Low | High (neural network) |
| **Best For** | Simple, discrete environments | Complex, high-dimensional environments |
| **Convergence** | Proven (with conditions) | Empirically works well |

### When to Use Each

**Use Q-Learning When**:
- State space is small (< 10,000 states)
- States are discrete
- Quick training needed
- Interpretability important
- Limited computational resources

**Use Deep Q-Learning When**:
- State space is large or continuous
- States are images or high-dimensional
- Generalization needed
- Willing to trade computational cost for performance
- Modern gaming or robotics applications

### Evolution Path

```
Q-Learning (1989)
    ↓
Deep Q-Network / DQN (2013)
    ↓
Double DQN (2015)
    ↓
Dueling DQN (2016)
    ↓
Prioritized Experience Replay (2016)
    ↓
Rainbow DQN (2017)
    ↓
Modern Variants (2018+)
```

---

## Practical Implementation Guide

### Network Architecture Design

#### Choosing Architecture for Different Inputs

**For Image Inputs** (Atari-style):
```python
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

**For Vector Inputs** (CartPole-style):
```python
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### Training Loop Structure

```python
def train_dqn(env, n_episodes, batch_size=32):
    # Initialize
    q_network = DQN(n_actions)
    target_network = DQN(n_actions)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.00025)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    epsilon = 1.0
    update_counter = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Select action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = q_values.argmax().item()
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(replay_buffer) > batch_size:
                # Sample batch
                batch = replay_buffer.sample(batch_size)
                
                # Compute loss
                loss = compute_loss(batch, q_network, target_network)
                
                # Update Q-network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network
                update_counter += 1
                if update_counter % 1000 == 0:
                    target_network.load_state_dict(q_network.state_dict())
            
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)
        
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
```

### Loss Computation

```python
def compute_loss(batch, q_network, target_network, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    
    # Current Q-values
    current_q = q_network(states).gather(1, actions.unsqueeze(1))
    
    # Compute target Q-values (Double DQN)
    with torch.no_grad():
        # Select actions using Q-network
        next_actions = q_network(next_states).argmax(1)
        # Evaluate using target network
        next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1))
        target_q = rewards + gamma * next_q * (1 - dones)
    
    # Compute loss
    loss = F.mse_loss(current_q, target_q)
    return loss
```

### Debugging and Monitoring

**Key Metrics to Track**:

```python
# 1. Episode rewards
episode_rewards = []

# 2. Average Q-values
avg_q_value = q_network(state).mean().item()

# 3. Loss values
losses = []

# 4. Epsilon values
epsilon_history = []

# 5. Win rate (for games)
win_count = 0
if episode_reward > threshold:
    win_count += 1
win_rate = win_count / episode

# Logging
wandb.log({
    'episode_reward': episode_reward,
    'avg_q_value': avg_q_value,
    'loss': loss.item(),
    'epsilon': epsilon,
    'win_rate': win_rate
})
```

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Not learning** | Flat reward curve | Check epsilon, increase exploration |
| **Unstable training** | Oscillating Q-values | Decrease learning rate, increase target update frequency |
| **Overestimation** | Q-values too high | Use Double DQN |
| **Forgetting** | Performance degrades | Increase replay buffer size |
| **Slow training** | Takes too long | Increase batch size, decrease target update C |
| **Exploding gradients** | NaN losses | Clip gradients, normalize inputs |

### Gradient Clipping

```python
# Add after loss.backward()
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10)
optimizer.step()
```

---

## Performance Benchmarks

### Atari Games Performance (DQN Paper)

| Game | Random Agent | Human | DQN |
|------|-------------|-------|-----|
| **Breakout** | 1.7 | 30.5 | 401.2 |
| **Pong** | -20.7 | 14.6 | 20.9 |
| **Space Invaders** | 148 | 1,668 | 1,976 |
| **Seaquest** | 68.4 | 42,054 | 5,286 |
| **Beam Rider** | 363.9 | 16,926 | 8,627 |

**Key Observations**:
- DQN surpasses human performance on some games
- Massive improvement over random agent
- Still room for improvement on complex games

### Training Time

**Typical Training Requirements**:
- **Episodes**: 1,000 to 10,000+ episodes
- **Frames**: 10 million to 50 million frames
- **Wall Time**: Hours to days on GPU
- **Hardware**: GPU recommended for image-based tasks

---

## Advanced Topics

### Improvements Beyond Basic DQN

#### 1. Prioritized Experience Replay

**Idea**: Not all experiences are equally important

**Implementation**:
```python
# Assign priority based on TD error
priority = abs(td_error) + epsilon
replay_buffer.push(experience, priority)

# Sample with probability proportional to priority
batch = replay_buffer.sample_prioritized(batch_size)
```

**Benefits**:
- Learn more from surprising experiences
- Faster convergence
- Better sample efficiency

#### 2. Dueling DQN Architecture

**Split Q-value into two components**:
```
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```
Where:
- V(s): Value of being in state s
- A(s,a): Advantage of action a in state s

**Benefits**:
- Better generalization
- More robust value estimates
- Especially helpful when many actions don't affect value

#### 3. Multi-step Returns

**Instead of 1-step TD**:
```
Q_target = r_t + γ r_{t+1} + γ² r_{t+2} + ... + γⁿ max Q(s_{t+n}, a)
```

**Benefits**:
- Faster credit assignment
- Better for sparse rewards
- Balance between MC and TD

#### 4. Noisy Networks

**Replace epsilon-greedy with learned exploration**:
- Add noise to network weights
- Network learns when/how to explore
- More sophisticated exploration

#### 5. Distributional RL

**Learn distribution of returns instead of expected value**:
- Capture uncertainty
- More robust to noise
- Better decision making

### Rainbow DQN

**Combines all improvements**:
1. Double DQN
2. Prioritized Replay
3. Dueling Networks
4. Multi-step Learning
5. Distributional RL
6. Noisy Networks

**Result**: State-of-the-art Atari performance

---

## Summary

### Key Takeaways

1. **Deep Q-Learning extends Q-Learning**
   - Uses neural networks instead of tables
   - Scales to large/continuous state spaces
   - Enables complex tasks like Atari games

2. **Core Architecture (DQN)**
   - Input: Stack of 4 preprocessed frames
   - Convolutional layers: Extract spatial features
   - Fully connected layers: Output Q-values
   - Training: Minimize loss via gradient descent

3. **Three Key Stabilization Techniques**
   - **Experience Replay**: Reuse experiences, break correlations
   - **Fixed Q-Target**: Separate target network for stability
   - **Double DQN**: Decouple selection and evaluation to reduce overestimation

4. **Training Process**
   - Sampling phase: Collect experiences
   - Training phase: Learn from random batches
   - Epsilon-greedy: Balance exploration/exploitation
   - Gradual improvement through iterations

5. **Practical Considerations**
   - Preprocessing essential (grayscale, resize, stack)
   - Hyperparameters matter (epsilon, learning rate, C)
   - Training takes significant time/computation
   - Monitoring crucial for debugging

### From Q-Learning to DQN Timeline

```
1989: Q-Learning (Watkins)
        ↓
2013: Deep Q-Network (Mnih et al., DeepMind)
      - First to play Atari from pixels
      - Human-level performance on some games
        ↓
2015: Double DQN (van Hasselt et al.)
      - Reduced overestimation
        ↓
2016: Dueling DQN (Wang et al.)
      - Better value estimation
        ↓
2016: Prioritized Experience Replay (Schaul et al.)
      - Sample important experiences more
        ↓
2017: Rainbow DQN (Hessel et al.)
      - Combines all improvements
        ↓
2018+: Continued refinements
```

### Impact and Applications

**Original Impact**:
- Proved deep RL could work on complex tasks
- Sparked explosion in deep RL research
- Showed path to general game-playing AI

**Current Applications**:
- Robotics control
- Resource management
- Game AI
- Recommendation systems
- Traffic control
- Energy optimization

### Limitations

1. **Sample Inefficiency**
   - Requires millions of frames
   - Expensive in real-world applications

2. **Discrete Actions Only**
   - Cannot handle continuous action spaces directly
   - Need policy gradient methods for continuous control

3. **Partial Observability**
   - Assumes fully observable state
   - Struggles with hidden information

4. **Exploration Challenges**
   - Epsilon-greedy is simple but not optimal
   - May struggle with sparse rewards

5. **Computational Cost**
   - Requires significant GPU resources
   - Training time can be days/weeks

### Future Directions

1. **Sample Efficiency**
   - Model-based methods
   - Better exploration strategies
   - Transfer learning

2. **Scalability**
   - Distributed training
   - Hierarchical RL
   - Multi-task learning

3. **Robustness**
   - Sim-to-real transfer
   - Adversarial robustness
   - Safe exploration

4. **General Intelligence**
   - Multi-game agents
   - Zero-shot generalization
   - Meta-learning

---

## References and Resources

### Foundational Papers

1. **DQN (2013)**
   - Mnih et al., "Playing Atari with Deep Reinforcement Learning"
   - Introduced Deep Q-Learning
   - arXiv:1312.5602

2. **DQN Nature Paper (2015)**
   - Mnih et al., "Human-level control through deep reinforcement learning"
   - Published in Nature
   - Demonstrated human-level Atari performance

3. **Double DQN (2015)**
   - van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
   - Addressed overestimation problem

4. **Dueling DQN (2016)**
   - Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
   - Split value and advantage streams

5. **Prioritized Experience Replay (2016)**
   - Schaul et al., "Prioritized Experience Replay"
   - Sample important transitions more

6. **Rainbow (2017)**
   - Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning"
   - Combined all major improvements

### Additional Reading

- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
- Deep RL Bootcamp (Berkeley)
- OpenAI Spinning Up documentation
- Stable-Baselines3 documentation

### Code Resources

- **Stable-Baselines3**: `pip install stable-baselines3`
- **RL-Zoo**: Pre-trained models and training scripts
- **OpenAI Gym**: Standard RL environments
- **Arcade Learning Environment**: Atari games

---

*These notes cover Unit 3: Deep Q-Learning with Atari Games from the Hugging Face Deep RL Course. Practice implementing DQN and experiment with different hyperparameters to solidify your understanding!*