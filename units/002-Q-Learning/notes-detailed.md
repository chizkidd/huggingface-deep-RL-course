# Unit 2: Introduction to Q-Learning

## Table of Contents

1. [Introduction](#1-introduction)
2. [What is RL? A Short Recap](#2-what-is-rl-a-short-recap)
3. [The Two Types of Value-Based Methods](#3-the-two-types-of-value-based-methods)
4. [The Bellman Equation](#4-the-bellman-equation)
5. [Monte Carlo vs Temporal Difference Learning](#5-monte-carlo-vs-temporal-difference-learning)
6. [Mid-way Recap](#6-mid-way-recap)
7. [Introducing Q-Learning](#7-introducing-q-learning)
8. [The Q-Learning Algorithm](#8-the-q-learning-algorithm)
9. [A Q-Learning Example](#8-a-q-learning-example)
10. [Q-Learning Recap](#9-q-learning-recap)
11. [Glossary](#10-glossary)

-----

## 1. Introduction

In Unit 2, we dive deeper into **value-based methods** in Reinforcement Learning and study our first RL algorithm: **Q-Learning**.

### What We’ll Learn

- Value-based methods and how they work
- The differences between Monte Carlo and Temporal Difference Learning
- Q-Learning algorithm implementation
- Training agents from scratch

### Practical Applications

We’ll implement a Q-Learning agent and train it in two environments:

1. **Frozen-Lake-v1**: Navigate from starting state (S) to goal state (G) by walking only on frozen tiles (F) and avoiding holes (H)
1. **Autonomous Taxi**: Learn to navigate a city to transport passengers from point A to point B

### Why This Matters

This unit is fundamental for understanding Deep Q-Learning, which was the first Deep RL algorithm to play Atari games and achieve human-level performance on games like Breakout and Space Invaders.

-----

## 2. What is RL? A Short Recap

### Core Concepts

#### The Agent’s Goal

In RL, we build agents that can **make smart decisions** through:

- **Interaction**: Learning from the environment through trial and error
- **Feedback**: Receiving rewards (positive or negative) as unique feedback
- **Optimization**: Maximizing expected cumulative reward (due to the reward hypothesis)

#### The Policy (π)

The agent’s **decision-making process** is called the policy:

- Given a state, the policy outputs an action or probability distribution over actions
- Given an observation of the environment, a policy provides the action(s) the agent should take

#### Finding the Optimal Policy (π*)

The goal is to find the **optimal policy π*** that leads to the best expected cumulative reward.

### Two Main RL Methods

#### 1. Policy-Based Methods

- **Approach**: Train the policy directly to learn which action to take given a state
- **Mechanism**: Direct mapping from state to action
- **No value function needed**

#### 2. Value-Based Methods

- **Approach**: Train a value function to learn which state is more valuable
- **Mechanism**: Use the value function to take actions that lead to valuable states
- **Indirect policy derivation**

**Note**: In this unit, we focus on value-based methods.

-----

## 3. The Two Types of Value-Based Methods

### Overview

In value-based methods, we **learn a value function** that maps a state to the expected value of being at that state.

The **value of a state** = expected discounted return the agent can get if it starts at that state and acts according to the policy.

### How Value-Based Methods Work

#### Policy Definition

- In value-based methods, we **don’t train a policy directly**
- Instead, we **specify the policy’s behavior** (e.g., Greedy Policy)
- The policy uses values from the value function to select actions

#### Key Difference from Policy-Based Methods

- **Policy-based**: Train the policy directly; the training defines behavior
- **Value-based**: Define policy behavior by hand; train a value function

**Example**: If we want a policy that always selects the action leading to the biggest reward, we create a **Greedy Policy** that uses the trained value function.

### The Two Types of Value Functions

#### 1. State-Value Function V(s)

**Definition**:

```
V^π(s) = Expected return if agent starts at state s and follows policy π forever
```

**Key Points**:

- Outputs expected return for each state
- Assumes agent follows the policy from that state onward
- Evaluates “how good is this state?”

**Example Use**:

- If a state has value -7, it means starting from that state and following the policy (e.g., right, right, right, down, down, right, right) yields an expected return of -7

#### 2. Action-Value Function Q(s,a)

**Definition**:

```
Q^π(s,a) = Expected return if agent starts in state s, takes action a, 
           then follows policy π forever
```

**Key Points**:

- Outputs expected return for each state-action pair
- Evaluates “how good is taking this action in this state?”
- More fine-grained than state-value function

**Difference Summary**:

- **State-value function**: Calculates value of a state S_t
- **Action-value function**: Calculates value of state-action pair (S_t, A_t)

### Computational Challenge

Both value functions require calculating **expected returns**, which means:

- Summing all rewards an agent can get starting from a state
- This can be computationally expensive

**Solution**: The Bellman Equation helps us solve this efficiently!

-----

## 4. The Bellman Equation

### Purpose

The Bellman equation **simplifies value estimation** by avoiding the need to calculate the entire sum of expected returns from scratch for each state.

### The Problem Without Bellman

To calculate V(S_t), we would need to:

1. Calculate the sum of expected rewards from that state
1. Follow the policy for all future timesteps
1. Repeat this process for every state

This creates redundant calculations!

### The Bellman Solution

#### Core Idea

Instead of calculating the complete expected return, we can express the value of any state as:

```
V(S_t) = Immediate reward R_{t+1} + Discounted value of next state (γ * V(S_{t+1}))
```

#### Recursive Nature

The Bellman equation is **recursive**:

- Value of current state = immediate reward + discounted value of next state
- Value of next state = its immediate reward + discounted value of its next state
- And so on…

### Mathematical Formulation

```
V(S_t) = R_{t+1} + γ * V(S_{t+1})
```

Where:

- **V(S_t)**: Value of current state
- **R_{t+1}**: Immediate reward received after taking action
- **γ**: Discount factor (gamma)
- **V(S_{t+1})**: Value of next state

### Example Walkthrough

Consider calculating V(State 1):

**Traditional Method**:

```
V(State 1) = R_1 + R_2 + R_3 + R_4 + ... (sum all future rewards)
```

**Bellman Method**:

```
V(State 1) = R_1 + γ * V(State 2)
V(State 2) = R_2 + γ * V(State 3)
...
```

**Simplified (γ = 1)**:

```
V(S_t) = R_{t+1} + V(S_{t+1})
```

### Understanding Gamma (γ)

The discount factor critically affects learning:

- **γ = 0 (very low)**: Agent only cares about immediate rewards (short-sighted)
- **γ = 0.99 (typical)**: Agent balances immediate and future rewards
- **γ = 1 (no discount)**: Agent values all future rewards equally
- **γ >> 1 (very high)**: Would overvalue future rewards (rarely used)

### Key Advantage

Instead of calculating each value as a complete sum of expected returns (long process), we calculate it as:

```
Immediate reward + Discounted value of next state
```

This makes computation much more efficient!

-----

## 5. Monte Carlo vs Temporal Difference Learning

### Overview

These are two different **strategies for training** value functions or policy functions. Both use experience to solve the RL problem, but differ in **when** they learn.

### Key Difference

- **Monte Carlo**: Uses an **entire episode** of experience before learning
- **Temporal Difference**: Uses only **one step** (S_t, A_t, R_{t+1}, S_{t+1}) to learn

-----

### Monte Carlo: Learning at the End of the Episode

#### Characteristics

- Waits until the **end of the episode**
- Calculates G_t (return) for the complete episode
- Uses G_t as target for updating V(S_t)
- Requires **complete episode** of interaction

#### The Process

1. **Start Episode**: Always start at the same starting point
1. **Take Actions**: Agent uses policy (e.g., Epsilon Greedy Strategy)
1. **Collect Experience**: Get rewards and next states
1. **Terminate Episode**: Based on conditions (e.g., goal reached, max steps exceeded)
1. **Calculate Return**: Sum total rewards G_t
1. **Update Value Function**:
   
   ```
   V(S_t) = V(S_t) + lr * [G_t - V(S_t)]
   ```
1. **Repeat**: Start new episode with updated knowledge

#### Example Scenario: Mouse and Cheese

**Setup**:

- Mouse explores environment with random actions
- Collects: state, action, reward, next_state tuples
- Episode ends after > 10 steps

**Calculation**:

```
G_0 = R_1 + R_2 + R_3 + ...
G_0 = 1 + 0 + 0 + 0 + 0 + 0 + 1 + 1 + 0 + 0 = 3

V(S_0) = V(S_0) + lr * [G_0 - V(S_0)]
V(S_0) = 0 + 0.1 * [3 - 0]
V(S_0) = 0.3
```

**Advantages**:

- Uses actual, complete returns
- More accurate for episodic tasks
- Unbiased estimates

**Disadvantages**:

- Must wait for episode to finish
- Cannot learn from incomplete episodes
- Slower learning for long episodes

-----

### Temporal Difference Learning: Learning at Each Step

#### Characteristics

- Waits for only **one interaction** (one step)
- Forms a TD target using: R_{t+1} + γ * V(S_{t+1})
- Updates V(S_t) at each step
- **Doesn’t need complete episode**

#### The Process

1. **Take One Step**: Execute action, observe reward and next state
1. **Estimate Return**: Use TD target instead of actual G_t
   
   ```
   TD target = R_{t+1} + γ * V(S_{t+1})
   ```
1. **Update Immediately**:
   
   ```
   V(S_t) = V(S_t) + lr * [R_{t+1} + γ * V(S_{t+1}) - V(S_t)]
   ```

#### Bootstrapping

TD is called **bootstrapping** because:

- It bases update on an **existing estimate** V(S_{t+1})
- Not on a complete sample G_t
- “Pulls itself up by its bootstraps”

#### TD(0) - One-Step TD

- Updates value function after **any individual step**
- Most basic form of TD learning

#### Example: Mouse Takes One Step

**Setup**:

- Learning rate = 0.1
- Discount rate = 1 (no discount)
- Mouse goes left and eats cheese

**Calculation**:

```
R_{t+1} = 1 (ate cheese)

New V(S_0) = V(S_0) + lr * [R_1 + γ * V(S_1) - V(S_0)]
New V(S_0) = 0 + 0.1 * [1 + 1 * 0 - 0]
New V(S_0) = 0.1
```

**Advantages**:

- Can learn before episode ends
- Works with continuing (non-episodic) tasks
- Generally faster learning
- Lower variance

**Disadvantages**:

- Uses estimates (bootstrapping)
- Can be biased initially
- Requires careful parameter tuning

-----

### Monte Carlo vs TD Summary

|Aspect            |Monte Carlo                |Temporal Difference               |
|------------------|---------------------------|----------------------------------|
|**Update Timing** |End of episode             |Each step                         |
|**Update Target** |Actual return G_t          |Estimated return (TD target)      |
|**Formula**       |V(S) = V(S) + α[G_t - V(S)]|V(S) = V(S) + α[R + γV(S’) - V(S)]|
|**Requirement**   |Complete episode           |Single step                       |
|**Bias**          |Unbiased                   |Initially biased                  |
|**Variance**      |High variance              |Lower variance                    |
|**Episode Type**  |Episodic only              |Both episodic & continuing        |
|**Learning Speed**|Slower                     |Faster                            |

-----

## 6. Mid-way Recap

### Value-Based Methods Summary

#### Two Types of Value Functions

1. **State-Value Function**
- Outputs expected return if agent starts at a given state
- Assumes following policy forever after
- Answers: “How good is this state?”
1. **Action-Value Function**
- Outputs expected return if agent starts in given state and takes given action
- Then follows policy forever after
- Answers: “How good is this action in this state?”

#### Policy in Value-Based Methods

- Rather than learning the policy directly
- We **define the policy by hand** (e.g., Greedy Policy)
- We **learn a value function**
- Optimal value function → Optimal policy

### Two Methods to Update Value Function

1. **Monte Carlo Method**
- Updates from complete episode
- Uses actual discounted return
- Formula: `V(S) = V(S) + α[G_t - V(S)]`
1. **TD Learning Method**
- Updates from single step
- Replaces unknown G_t with TD target
- Formula: `V(S) = V(S) + α[R + γV(S') - V(S)]`

### Key Takeaway

Both methods aim to learn value functions, but differ in:

- **When** they update (end of episode vs. each step)
- **What** they use as target (actual return vs. estimated return)

-----

## 7. Introducing Q-Learning

### What is Q-Learning?

Q-Learning is an **off-policy value-based method** that uses a **TD approach** to train its action-value function.

**Key Characteristics**:

- **Off-policy**: Different policies for acting and updating
- **Value-based**: Finds optimal policy indirectly via value function
- **TD approach**: Updates after each step, not episode end

### The Q-Function

#### Definition

Q-Learning trains a **Q-function** (action-value function) that:

- Determines value of being at a particular state
- Taking a specific action at that state
- Returns the **Q-value** (quality value)

**Mathematical Notation**:

```
Q(s, a) = Expected cumulative reward from state s, taking action a
```

#### Q vs “Quality”

The **Q** stands for **“Quality”**:

- Quality (value) of taking that action in that state
- Higher Q-value = Better action choice

### Value vs Reward

**Important Distinction**:

|Concept   |Definition                                        |Timing        |
|----------|--------------------------------------------------|--------------|
|**Value** |Expected cumulative reward from state/state-action|Future-looking|
|**Reward**|Feedback from environment after action            |Immediate     |

- **Value**: What we expect to get in total
- **Reward**: What we just received

### The Q-Table

#### Structure

The Q-function is encoded by a **Q-table**:

- Table where each cell = state-action pair value
- Think of it as the “memory” or “cheat sheet” of Q-function

#### Example: Simple Maze

**Maze Setup**:

- 2 x 3 grid (6 positions)
- Mouse can move: up, down, left, right

**Q-Table Structure**:

```
State (Position) | Action: Up | Action: Down | Action: Left | Action: Right
----------------|-----------|--------------|--------------|---------------
Position 1      |     0     |      0       |      0       |       0
Position 2      |     0     |      0       |      0       |       0
...             |    ...    |     ...      |     ...      |      ...
```

Initially, all values = 0 (or random initialization)

#### How Q-Table is Used

1. Agent observes current state
1. Q-function searches Q-table for that state row
1. Returns Q-values for all possible actions
1. Policy selects action based on Q-values

### Q-Learning Process Overview

**Training Cycle**:

1. Start with useless Q-table (arbitrary/zero values)
1. Agent explores environment
1. Update Q-table based on experience
1. Q-table progressively approximates optimal values
1. Converge to optimal Q-function → optimal policy

**Result**:

```
Optimal Q-function → Optimal Q-table → Optimal Policy
```

**Why?**

- Optimal Q-table tells us best action for each state
- We know the value of each state-action pair
- We can always choose the best action

-----

## 8. The Q-Learning Algorithm

### Pseudocode Overview

The algorithm consists of 4 main steps repeated over multiple episodes:

```
1. Initialize Q-table
2. Choose action using epsilon-greedy
3. Perform action, get reward and next state  
4. Update Q(S_t, A_t)
```

Let’s examine each step in detail:

-----

### Step 1: Initialize the Q-Table

**Process**:

- Create Q-table with dimensions: [number of states] x [number of actions]
- Initialize all Q-values (usually to 0)

**Example**:

```
Q-table (initial):
         Left  Right  Up  Down
State1    0     0     0    0
State2    0     0     0    0
State3    0     0     0    0
...
```

**Why zero?**

- No prior knowledge about environment
- Agent will learn through exploration
- Other initialization strategies exist (random, optimistic)

-----

### Step 2: Choose Action Using Epsilon-Greedy Strategy

#### The Epsilon-Greedy Strategy

**Purpose**: Handles exploration/exploitation trade-off

**How it Works**:

```
With probability (1 - ε): EXPLOITATION - choose best action (highest Q-value)
With probability ε:       EXPLORATION - choose random action
```

#### Parameters

- **ε (epsilon)**: Probability of random exploration
- Starts high (e.g., ε = 1.0): Maximum exploration at beginning
- Decays over time: Gradually shift toward exploitation

#### Implementation Logic

```python
if random() < epsilon:
    # Exploration: random action
    action = random_action()
else:
    # Exploitation: best known action
    action = argmax(Q[state, :])
```

#### Why This Works

**Early Training** (ε high, e.g., 0.9):

- 90% exploration, 10% exploitation
- Discover new states and actions
- Build knowledge about environment

**Late Training** (ε low, e.g., 0.1):

- 10% exploration, 90% exploitation
- Use learned knowledge
- Still some exploration to prevent local optima

#### Epsilon Decay

```python
epsilon = epsilon * decay_rate
# or
epsilon = max(min_epsilon, epsilon * decay_rate)
```

-----

### Step 3: Perform Action A_t, Get Reward R_{t+1} and Next State S_{t+1}

**Process**:

1. Execute chosen action in environment
1. Observe consequences:
- Immediate reward R_{t+1}
- New state S_{t+1}
- Whether episode terminated

**Example**:

```
Current state: S_t = "Position 1"
Action: A_t = "Go Right"
→ Execute action in environment
Result:
  - Reward: R_{t+1} = +1 (picked up small cheese)
  - Next state: S_{t+1} = "Position 2"
  - Done: False (episode continues)
```

-----

### Step 4: Update Q(S_t, A_t)

#### The Update Formula

**TD Learning Update**:

```
Q(S_t, A_t) = Q(S_t, A_t) + α[R_{t+1} + γ * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Components**:

- **Q(S_t, A_t)**: Current Q-value we’re updating
- **α (alpha)**: Learning rate (e.g., 0.1)
- **R_{t+1}**: Immediate reward received
- **γ (gamma)**: Discount factor (e.g., 0.99)
- **max_a Q(S_{t+1}, a)**: Maximum Q-value in next state (over all actions)

#### Breaking Down the Formula

**TD Target**:

```
TD_target = R_{t+1} + γ * max_a Q(S_{t+1}, a)
```

This is our estimate of true Q-value

**TD Error**:

```
TD_error = TD_target - Q(S_t, A_t)
```

Difference between estimate and current value

**Update**:

```
New Q(S_t, A_t) = Old Q(S_t, A_t) + α * TD_error
```

Move current value toward target

#### How to Form TD Target

1. **Get immediate reward** R_{t+1}
1. **Find best next action**:
- Look at all actions in next state S_{t+1}
- Select action with maximum Q-value
- Use **greedy policy** (not epsilon-greedy!)
1. **Combine**: R_{t+1} + γ * max Q-value

**Note**: This is **off-policy** because:

- **Acting policy**: Epsilon-greedy (exploration + exploitation)
- **Update policy**: Greedy (pure exploitation for TD target)

#### Example Calculation

**Scenario**:

- Current state: S_t, Action: A_t
- Reward received: R_{t+1} = 1
- Next state: S_{t+1}
- Current Q(S_t, A_t) = 0
- Learning rate: α = 0.1
- Discount: γ = 0.99
- Q-values in S_{t+1}: [0, 0, 0.5, 0] → max = 0.5

**Calculation**:

```
TD_target = 1 + 0.99 * 0.5 = 1.495
TD_error = 1.495 - 0 = 1.495
Q(S_t, A_t) = 0 + 0.1 * 1.495 = 0.1495
```

-----

### Off-Policy vs On-Policy

#### Off-Policy (Q-Learning)

**Definition**: Uses different policies for:

- **Acting** (behavior policy): Epsilon-greedy
- **Updating** (target policy): Greedy

**Why?**

- Act with exploration (epsilon-greedy) to discover environment
- Update assuming optimal behavior (greedy) for faster convergence

**Advantage**: Can learn optimal policy while following exploratory policy

#### On-Policy (e.g., SARSA)

**Definition**: Uses same policy for both:

- **Acting**: Epsilon-greedy
- **Updating**: Epsilon-greedy

**Algorithm**:

```
Q(S_t, A_t) = Q(S_t, A_t) + α[R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Key Difference**: Next action A_{t+1} selected by same epsilon-greedy policy

**Advantage**: Learns the value of policy actually being followed

#### Comparison Table

|Aspect           |Off-Policy (Q-Learning)|On-Policy (SARSA)          |
|-----------------|-----------------------|---------------------------|
|**Acting Policy**|Epsilon-greedy         |Epsilon-greedy             |
|**Update Policy**|Greedy                 |Epsilon-greedy             |
|**Update Uses**  |max_a Q(S’, a)         |Q(S’, A’) where A’ ~ policy|
|**Learns**       |Optimal policy         |Current policy             |
|**Convergence**  |To optimal             |To policy being followed   |
|**Safety**       |May learn risky optimal|Learns safe actual policy  |

-----

## 9. A Q-Learning Example

Let’s walk through a complete example step by step.

### Environment Setup

**The Maze**:

- Simple grid world with mouse
- **Goal**: Eat the big pile of cheese (bottom right)
- **Avoid**: Poison
- **Start**: Always same starting point

**Episode Termination**:

- Eat poison → Episode ends
- Eat big cheese → Episode ends
- Take > 5 steps → Episode ends

**Parameters**:

- Learning rate (α): 0.1
- Discount rate (γ): 0.99

### Reward Function

|Action Result      |Reward|
|-------------------|------|
|Move to empty state|0     |
|Small cheese       |+1    |
|Big cheese (goal)  |+10   |
|Poison (death)     |-10   |
|More than 5 steps  |0     |

### Goal

Train agent to learn optimal policy: **Right → Right → Down**

-----

### Training Walkthrough

#### Initial State: Step 1

**Step 1: Initialize Q-Table**

```
Q-table (all zeros):
         Left  Right  Up  Down
Start     0     0     0    0
Middle    0     0     0    0
Goal      0     0     0    0
```

Q-table is useless initially - needs training!

-----

#### Training Timestep 1

**Step 2: Choose Action**

- Epsilon = 1.0 (100% exploration)
- Result: Random action
- **Action chosen**: Go Right

**Step 3: Perform Action**

- Start state → Move right
- Outcome:
  - Reward: R_{t+1} = +1 (small cheese)
  - Next state: Middle position
  - Episode continues

**Step 4: Update Q-Value**

**Current values**:

- Q(Start, Right) = 0
- Best Q-value in next state = 0 (all zeros)
- Reward = 1
- α = 0.1, γ = 0.99

**Calculation**:

```
Q(Start, Right) = Q(Start, Right) + α[R + γ * max Q(next state) - Q(Start, Right)]
Q(Start, Right) = 0 + 0.1[1 + 0.99 * 0 - 0]
Q(Start, Right) = 0 + 0.1[1]
Q(Start, Right) = 0.1
```

**Updated Q-table**:

```
         Left  Right  Up  Down
Start     0    0.1    0    0    ← Updated!
Middle    0     0     0    0
Goal      0     0     0    0
```

-----

#### Training Timestep 2

**Step 2: Choose Action**

- Epsilon = 0.99 (still mostly exploring)
- Result: Random action
- **Action chosen**: Go Down ← Not optimal!

**Step 3: Perform Action**

- Middle position → Move down
- Outcome:
  - Reward: R_{t+1} = -10 (ate poison!)
  - Next state: Death state
  - Episode terminates

**Step 4: Update Q-Value**

**Current values**:

- Q(Middle, Down) = 0
- Reward = -10
- Episode ended (no next state to consider)
- α = 0.1, γ = 0.99

**Calculation**:

```
Q(Middle, Down) = Q(Middle, Down) + α[R - Q(Middle, Down)]
Q(Middle, Down) = 0 + 0.1[-10 - 0]
Q(Middle, Down) = -1
```

**Updated Q-table**:

```
         Left  Right  Up  Down
Start     0    0.1    0    0
Middle    0     0     0   -1    ← Updated! (negative = bad)
Goal      0     0     0    0
```

-----

### Key Observations

**After Just 2 Timesteps**:

1. Agent learned going Right from Start gets reward (Q = 0.1)
1. Agent learned going Down from Middle is bad (Q = -1)

**Progress**:

- Q-table becoming more informative
- Agent getting “smarter” about environment
- Still needs many more iterations to converge

**Next Steps**:

- Continue episodes
- Epsilon gradually decreases
- Q-values converge to optimal
- Agent learns optimal path: Right → Right → Down

### Complete Training Process

**As training continues** (hundreds/thousands of episodes):

1. **Exploration phase** (high ε):
- Try different actions
- Discover rewards/penalties
- Build Q-table knowledge
1. **Exploitation phase** (low ε):
- Use learned Q-values
- Take better actions
- Refine Q-value estimates
1. **Convergence**:
- Q-values stabilize
- Optimal policy emerges
- Agent consistently succeeds

**Final Result**:

```
Optimal Q-table → Best action at each state → Optimal policy
```

-----

## 10. Q-Learning Recap

### What is Q-Learning?

Q-Learning is the RL algorithm that:

1. **Trains a Q-function** (action-value function)
- Encoded internally by Q-table
- Contains all state-action pair values
1. **Uses Q-table for decisions**
- Given state and action
- Searches Q-table for corresponding value
1. **Converges to optimal policy**
- Optimal Q-function → Optimal Q-table
- Optimal Q-table → Know best action per state
- Best action per state → Optimal policy

### The Training Journey

**Initial State**:

- Q-table is useless (arbitrary/zero values)
- Agent has no knowledge

**During Training**:

- Agent explores environment
- Updates Q-table incrementally
- Q-table becomes better approximation

**Final State**:

- Optimal Q-function achieved
- Optimal policy derived
- Agent can act optimally

### The Q-Learning Algorithm

**Complete Pseudocode**:

```
Initialize Q-table Q(s,a) arbitrarily for all s,a

For each episode:
    Initialize state S
    
    For each step of episode:
        Choose A from S using epsilon-greedy policy
        
        Take action A, observe R, S'
        
        Q(S,A) ← Q(S,A) + α[R + γ * max_a Q(S',a) - Q(S,A)]
        
        S ← S'
        
    Until S is terminal
```

**Key Points**:

- Off-policy algorithm
- TD learning (updates each step)
- Uses epsilon-greedy for exploration
- Converges to optimal policy

-----

## 11. Glossary

### Main Concepts

#### Strategies to Find Optimal Policy

**Policy-Based Methods**

- Train policy directly with neural network
- Network outputs action to take given state
- Network adjusts based on experience
- Direct state → action mapping

**Value-Based Methods**

- Train value function to output state/state-action values
- Value function doesn’t define actions directly
- Must specify agent behavior based on values
- Example: Greedy policy uses values to select actions
- Indirect policy through value function

#### Value Function Types

**State-Value Function**

- For each state: expected return starting from that state
- Assumes agent follows policy until end
- Notation: V^π(s)
- Answers: “How good is this state?”

**Action-Value Function**

- For each state-action pair: expected return
- Starting from state, taking that action
- Then following policy forever
- Notation: Q^π(s,a)
- Answers: “How good is this action in this state?”

### Exploration Strategies

#### Epsilon-Greedy Strategy

- **Purpose**: Balance exploration and exploitation
- **With probability (1-ε)**: Choose action with highest Q-value (exploitation)
- **With probability ε**: Choose
