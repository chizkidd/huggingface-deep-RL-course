# Unit 2: Introduction to $Q$-Learning

This unit explores the foundations of value-based reinforcement learning. While Unit 1 introduced Deep RL using neural networks, Unit 2 focuses on tabular methods where we learn to estimate the value of states and actions manually using the Bellman Equation.

## 2.1 Introduction

>In this unit, we move from Policy-Based methods (where we learn a policy function directly) to **Value-Based methods**. The core objective is to create a **$Q$-Table**, a cheat sheet that tells the agent the maximum expected future reward for every possible action in every possible state.

This unit dives deeper into **value-based RL methods** and introduces **$Q$-Learning**, one of the foundational RL algorithms. It explains theoretical concepts and then applies them to simple environments such as `FrozenLake-v1` and an autonomous taxi. This unit also sets up the conceptual foundation for later units on Deep $Q$-Learning and `Atari` game agents. 

Key learning outcomes:
- Create a **$Q$-Table.**
- Value-based methods and how they differ from policy-based approaches.
- Contrast between Monte Carlo and Temporal Difference learning.
- Understanding and implementing the $Q$-Learning algorithm from scratch. 


## 2.2 What is RL? A Short Recap

Reinforcement Learning (RL) focuses on training an **agent** to make decisions through interaction with an **environment**. The agent learns by receiving **rewards** for its actions and aims to **maximize cumulative reward** over time. The policy, denoted $\epsilon$, defines the agent’s decision strategy. The optimal policy $\pi^\*$ leads to the best possible long-term returns. 

 ![](https://sijunhe.github.io/assets/images/posts/rlhf/basic_rl.jpeg)

Two broad classes of RL methods:
- **Policy-based methods**: Directly learn a policy that maps states to actions.
- **Value-based methods**: Learn a value function that approximates future expected return, and derive a policy from that value function.
   - In Value-Based RL, the agent learns a **Value Function** that maps a state (or state-action pair) to a value.
      - **The Goal:** Find the optimal value function $V^\*(s)$ or $Q^\*(s, a)$.
      - **The Logic:** If we know the value of every state, the optimal policy $\pi^\*$ is simply to always take the action that leads to the state with the highest value.

## 2.3 Two Types of Value-Based Methods
In value-based approaches, a policy is derived by selecting actions that maximize value (for example, greedy or $\epsilon$-greedy)

1. **State-Value Function $V(s)$:** Calculates the expected return if the agent starts in state $s$ and follows a given policy thereafter. It describes how good it is to be in a given state when following policy $\pi$.

$$V_{\pi}(s) = E_{\pi} [G_t | S_t = s] = \mathbb{E}[G_t \mid S_t = s, \pi]$$


2. **Action-Value Function $Q(s, a)$:** Calculates the expected return if the agent is in state $s$, takes action $a$, and then follows the policy thereafter. $Q$-values allow evaluating not just states, but specific action choices in states. This is what we use in $Q$-Learning.

$$Q_{\pi}(s, a) = E_{\pi} [G_t | S_t = s, A_t = a] = \mathbb{E}[G_t \mid S_t = s, A_t = a, \pi]$$


## 2.4 The Bellman Equation

The Bellman Equation is the mathematical foundation of RL. It provides a recursive decomposition of value functions & simplifies the calculation of the value function by breaking the expected return into two parts: the immediate reward plus the discounted value of the next state. Rather than summing all future rewards to compute value, the Bellman equation expresses value as:
- **Immediate reward** + **discounted value of successor state**.

For a state value: 

$$V(s) \leftarrow R + \gamma V(s')$$

$$V(s) = \mathbb{E}[ R_{t+1} + \gamma V(s_{t+1}) ]$$
  
This recursion simplifies iterative computation and underpins dynamic programming and many RL algorithms.

Where:

* $R$: Immediate reward.
* $\gamma$: Discount factor (importance of future rewards).
* $V(s')$: Value of the next state.

The discount factor $\gamma$ adjusts how much future rewards affect current value. Values close to 1 place more emphasis on long-term reward.


## 2.5 Monte Carlo (MC) vs. Temporal Difference (TD)

These are the two ways or learning strategies we use to update our value functions: 
### Monte Carlo (MC)
- Updates values **only after an entire episode** completes.
- Uses actual episode returns as targets.  
- ***Advantage:*** accurate targets.  
- ***Limitation:*** must wait until end of episode.

### Temporal Difference (TD) Learning
- Updates value estimates **after each step**.
- Uses **bootstrapping**: uses estimated value of next state instead of full return.
- More incremental and efficient than MC.
- `TD(0)` specifically updates after one time step using:
   - `TD target = immediate reward + γ $\gamma$ × estimated value of next state.`
     
| Feature | Monte Carlo (MC) | Temporal Difference (TD) |
| --- | --- | --- |
| **Learning** | At the end of the episode. | At every time step (Online). |
| **Requirement** | Needs complete episodes. | Works with incomplete episodes. |
| **Update Base** | Actual total return  $G_t$. | Estimated return $R_{t+1} + \gamma Q(s', a')$. |
| **Bias/Variance** | High variance, zero bias. | Low variance, some bias. |

***<u>Note:</u> $Q$-Learning uses Temporal Difference (TD) learning.***

## 2.6 Mid-way Recap
Summary of key points before $Q$-Learning:
- We want to find the optimal policy by finding the optimal value function.
- The value function represents the expected future discounted rewards.
- Value functions estimate expected returns for states or state-action pairs.
- We use the Bellman Equation to define the value of a state as the sum of immediate reward + discounted future values.
- Value-based methods do not directly train a policy, but ***derive one from estimates.***
- Monte Carlo and TD are two main strategies for learning value functions.
- TD learning allows us to update our estimates at every step without waiting for the episode to end unlike MC learning which updates our estimates only at the end of the episode.

## 2.7 $Q$-Learning: The Algorithm

**$Q$-Learning** is a model-free, **off-policy, value-based** RL algorithm that uses a **TD approach** to learn optimal action-value function $Q^\*(s, a)$. It uses a **$Q$-Table** to store the $Q$-values for all state-action pairs $(s, a)$. $Q$-Learning is **off-policy** because the "acting policy" (how the agent moves, usually $\epsilon$-greedy) is **different** from the "learning policy" (which assumes the agent will take the absolute best action in the next step). The $Q$-value represents the quality of taking action $a$ in state $s$ — the expected return following that choice.


### Key characteristics:
- **Off-policy**:
   - learning uses a different policy for action selection ($\epsilon$-greedy) and target selection (greedy).
   - learns optimal policy independent of agent’s actions. 
- **TD learning:** updates values step by step.  
- **$Q$-table:** stores learned estimates of action values $Q(s, a)$ for all state-action pairs.

### Exploration vs Exploitation  
A common strategy is **epsilon-greedy**:

- With probability $\epsilon$: **explore**
- With probability $1 - \epsilon$: **exploit** best current estimate  

Start with high $\epsilon$ and **decay** it over time.

The algorithm consists of:
1. Initialize $Q$-table with defaults (often zeros).
2. Select actions using **$\epsilon$-greedy policy** to balance exploration and exploitation.
3. Take action, observe reward and next state.
4. Update the corresponding $Q$-value using:

**The $Q$-Learning Update Rule (Bellman Optimality Equation]:**
$Q$-values are updated using the Bellman optimality principle:
- **Immediate reward** + **discounted value of the greedy estimate of the best future value**.

$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(s_{t+1},a) - Q(s_t,a_t)\bigr]$$

* **$\alpha$ (Learning Rate):** How much we update our value.
* **$\gamma$ (Discount Factor):** How much we care about future rewards.
* **$R_{t+1}$:** Immediate reward  
* **$\max_{a'} Q(s', a')$:** The greedy estimate of the best future value.

_Note: Off-policy means that the update uses the greedy action for the next state even if the action executed was exploratory._

## 2.8 $Q$-Learning Example

### Gridworld Case
In typical examples (e.g., grid worlds like `FrozenLake`):

1. Initialize $Q$-table with zeros  
2. For each episode:
   - Observe current state
   - Select an action via $\epsilon$-greedy (sometimes explore randomly, sometimes pick the best known action).
   - Execute action → observe next state $s'$ & reward $R$
   - Update $Q(s, a)$ (specific cell in the $Q$-Table) using TD formula
   - Update $\epsilon$ (decay)
3. Repeat until convergence ($1000$s of episodes)

Over time, $Q$-table values reflect the **expected cumulative return** for each state-action pair.

Once training ends, the optimal policy selects the **action with highest $Q$-value** in each state. 


### Maze Scenario
In a simple maze scenario:
- The agent starts with a $Q$-table of zeros.
- Using $\epsilon$-greedy selection, it explores actions and updates $Q$-values according to observed rewards and estimated future rewards.
- As training progresses, the $Q$-table approximates optimal values and reveals the best actions.

Example rewards:
- Slight positive ($>0$) for safe states, large positive ($>>>0$ for optimal goal, large negative ($<<<0$) for harmful states. 

Iterative updates gradually shape the policy toward maximum total reward.

## 2.9 $Q$-Learning Recap

$Q$-Learning trains an **action-value function** using a tabular representation. It:
- Uses TD updates to iteratively improve $Q$-value estimates.
- Learns optimal policy by selecting actions with highest $Q$-values after training.
- Starts with arbitrary values and converges as exploration and updates proceed.

Once the $Q$-table approximates $Q^\*$, the greedy policy derived from it approximates the optimal policy. 

### Key Terms
* **$Q$-Table:** A matrix where rows are states and columns are actions.
* **Exploration vs. Exploitation:** We use $\epsilon$-greedy to ensure the agent doesn't get stuck in local optima.
* **Target:** $R + \gamma \max Q(s', a')$.
* **TD Error:** The difference between the target and the current $Q(s, a)$.
 
## 2.10 Glossary
* **Policy-based methods:** Directly learn a policy 
* **Value-based methods:** Learn a value function and derive a policy from it. Methods that find the optimal policy by learning the value of states.
* **$Q$-Value:** The "Quality" or expected reward of taking a specific action in a specific state.
* **State-value function**: Expected return from a state following a policy. 
* **Action-value function ($Q$-function)**: Expected return from a state-action pair following a policy.
* **Monte Carlo (MC):**  Updates values only after an entire episode ends.
* **Temporal Difference (TD):** Updating an estimate based on another estimate.
* **$\epsilon$-greedy strategy:** An action selection policy/method that balances exploration and exploitation by picking a random action with probability $\epsilon$ and the best action with probability $1-\epsilon$.
* **Greedy strategy**: Always choose action with highest estimated value (exploitation only). 
* **Convergence:** When the $Q$-Table values stop changing significantly, meaning the agent has found the optimal strategy.
* **Off-policy vs On-policy**: Off-policy uses different policies for action selection during training versus evaluation; on-policy uses the same policy. 
* **Monte Carlo vs Temporal Difference**: MC updates after an episode; TD updates step-by-step using bootstrapped estimates.

## 2.11 Summary  
Unit 2 builds the foundation of value-based reinforcement learning:

- Understand value functions & Bellman equations  
- Introduced MC vs TD learning  
- Studied **$Q$-Learning**, the canonical value-based algorithm  
- Provided practical strategies like epsilon-greedy exploration

This prepares you for **Deep $Q$-Learning** and other advanced RL methods in the next unit.

