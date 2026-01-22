# Unit 3: Deep Q-Learning (Deep Reinforcement Learning)

This unit summarizes the key concepts from **Unit 3 of the Deep Reinforcement Learning course** on Hugging Face, covering Deep Q-Learning, Deep Q-Networks, the Deep Q-Learning Algorithm, and essential glossary terms.  

We explore the transition from Tabular RL to Deep Reinforcement Learning by introducing the **Deep Q-Network (DQN)**. We shift from using a simple lookup table to using neural networks as function approximators to handle environments with massive or continuous state spaces.

---

## 3.1 Introduction to Deep Q-Learning

In previous units, we used **Q-Learning**, which creates a Q-table where every row is a state and every column is an action. However, this becomes impossible when:

* **State spaces are huge:** For example, in Atari games, the number of possible pixel combinations is astronomical.
* **States are continuous:** You cannot create a table for infinite decimal values.

**Deep Q-Learning** solves this by replacing the Q-table with a **Neural Network** (the Q-Network) that learns to map a state to the Q-values of all possible actions.

### The Limitation of Tabular Q-Learning 
* **Scalability:** In environments with massive state spaces (like Atari games with pixels), creating a Q-table is computationally impossible.
  * *Example:* For a game with $210 \times 160$ pixels and 3 RGB channels, the number of states is $256^{(210 \times 160 \times 3)}$, which far exceeds the atoms in the observable universe. 
* **Lack of Generalization:** Tabular RL cannot "guess" the value of a new state; it must visit every state to learn its value. 
---

## 3.2 From Q-Learning to Deep Q-Learning

The core difference lies in how we represent the Q-function. To solve the scalability problem, we replace the Q-table with a **Function Approximator**—specifically a **Deep Neural Network**.

### The Temporal Difference (TD) Foundation
The transition relies on the TD update logic. 
- For state values: $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$
- For Q-values (Tabular): $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### The Problem with Tabular Q-Learning

In a table, we update individual cells using the Bellman equation. In Deep RL, we want to find a function Q(s, a; \theta)$ (where $\theta$ represents the weights of the network) that approximates the optimal Q-values.

### Why use Deep Learning? 
Neural networks are excellent at identifying patterns in high-dimensional data. This allows the agent to **generalize**: if it sees a state similar to one it has seen before, it can infer the correct action without having visited that exact state. 

### The Deep Q-Network (DQN) Architecture
Instead of a table, the state $s$ is fed into a neural network that outputs a vector of Q-values $[Q(s, a_1), Q(s, a_2), \dots, Q(s, a_n)]$. 


* **Input:** The state $s$ (e.g., 4 frames of a game to capture motion).
* **Hidden Layers:** Convolutional layers (if the input is visual) or Fully Connected layers.
* **Output:** A vector of Q-values $[Q(s, a_1), Q(s, a_2), \dots, Q(s, a_n)]$, one for each possible action in that state.

---

## 3.3 The Deep Q-Network (DQN) Components

To make training stable and successful, DQN introduces two critical concepts:

### 1. Experience Replay

In RL, consecutive experiences are highly correlated (e.g., frame $t$ is very similar to frame $t+1$). This "sequential" learning breaks the requirement of Deep Learning that data be **Independent and Identically Distributed (IID)**. 

Instead of learning from transitions as they happen (which are highly correlated), we store transitions $(s, a, r, s')$ in a **Replay Buffer**.

**The Solution:** 
1. Store transitions $(s, a, r, s', done)$ in a **Replay Buffer**. 
2. During training, sample a **random batch** of experiences from this buffer. 
3. **Benefit:** Reduces correlation between consecutive experiences and allows the network to "re-learn" from past experiences. This breaks the correlation between samples and allows the agent to learn from the same experience multiple times. 



### 2. Fixed Q-Targets
In Q-learning, we update our value towards a target:
$$Target = R + \gamma \max_{a'} Q(s', a')$$

In basic Deep Q-learning, the same network weights ($\theta$) are used for both the **prediction** and the **target**. This is like a cat chasing its own tail—every time we update the weights to get closer to the target, the target itself moves. In Deep RL, if we use the same network to predict the value and calculate the target, the "target" moves every time we update the weights, leading to oscillations.

**The Solution:** Use two networks:
1.  **The Q-Network (Online Network):** Trained to update weights at every step.
2.  **The Target Network:** A copy of the Q-network used to calculate the target. Its weights ($\theta^-$) are frozen and only updated to match the Online Network every $C$ steps.


---
## 3.4 The Deep Q-Algorithm
The training loop for DQN follows these steps:

1.  **Initialize** the Replay Buffer $B$, the Online Network ($Q_\theta$), and the Target Network ($Q_{\theta^-}$).
2.  **Choose Action:** Use an $\epsilon$-greedy policy to select an action $a$.
3.  **Execute:** Perform action $a$, observe reward $r$ and new state $s'$.
4.  **Store:** Save the transition $(s, a, r, s', done)$ in the Replay Buffer.
5.  **Sample:** Take a random batch of transitions from the buffer.
6.  **Calculate Target:**
    * If the state is terminal: $Y_i = r$
    * Otherwise: $Y_i = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$
7.  **Loss Function:** Compute the Mean Squared Error (MSE) loss:
    $$Loss = (Y_i - Q_\theta(s, a))^2$$
8.  **Optimize:** Perform Gradient Descent to minimize the loss and update the Online Network $\theta$.
9.  **Update Target:** Every $C$ steps, copy weights $\theta \to \theta^-$.
10.  **Repeat** steps 2 to 9 until convergence


---




## 3.5 Glossary

| Term | Definition |
| --- | --- |
| **Deep Q-Learning** | Uses neural networks to approximate Q-values for environments with large state spaces. |
| **Deep Q-Network (DQN)** | A neural network that predicts Q(s, a) for all possible actions given a state input. |
| **Experience Replay** | Memory storing transitions (s, a, r, s′) used for random mini-batch training to reduce correlation and improve efficiency. |
| **Target Network** | A separate neural network used to compute stable target Q-values, periodically synced with the main network. |
| **Double DQN** | Variation of DQN that uses two networks to reduce Q-value overestimation. |
| **Temporal Limitation** | Single frames lack motion information; hence, multiple frames are stacked for DQN input. |
| **Function Approximator** | A statistical tool/model (like a Neural Network) used to estimate a function when the input space is too large for a table. |
| **Exploration-Exploitation Tradeoff** | The balance between trying new actions ($\epsilon$) and choosing the best-known action. |
| **Temporal Difference (TD) Error** | The difference between the estimated Q-value and the actual reward plus discounted future reward. |
| **Catastrophic Forgetting** | A phenomenon where a network forgets old information as it learns new information; mitigated by Experience Replay. |
| **Dead ReLU** | A common problem in Deep RL where neurons "die" and always output zero; often solved using Leaky ReLU or lower learning rates. |
| **$\epsilon$-greedy Policy** | A policy where the agent chooses a random action with probability $\epsilon$ and the best action with probability $1-\epsilon$. | 
| **Bellman Equation** | The fundamental recursive equation that defines the value of a state or state-action pair. | 
| **Huber Loss** | A loss function often used in DQN that is less sensitive to outliers than MSE. |

---

