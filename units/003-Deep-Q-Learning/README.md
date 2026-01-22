# Unit 3: Deep Q-Learning (Deep Reinforcement Learning)

This document summarizes the key concepts from **Unit 3 of the Deep Reinforcement Learning course** on Hugging Face, covering Deep Q-Learning, Deep Q-Networks, the Deep Q-Learning Algorithm, and essential glossary terms.  

---

## 3.1 — Introduction (Deep Q-Learning)

Reinforcement learning (RL) agents learn to make decisions by interacting with an environment to **maximize cumulative reward**.  
Traditional **Q-Learning**, which uses a Q-table to store action-value estimates Q(s, a), works well for small discrete state spaces. 

However:

- Many real-world environments (e.g., Atari games) have **vast or continuous state spaces** (e.g., images) that make Q-tables impractical. 
- **Deep Q-Learning** replaces the Q-table with a **neural network** that approximates the action-value function Q(s, a) by learning to map a state to the Q-values of all possible actions/

This allows RL agents to scale to complex problems like playing Atari games from raw pixel input.

---

## 3.2 — From Q-Learning to Deep Q-Learning

### Q-Learning Recap

Q-Learning is a **model-free, value-based RL algorithm** that learns optimal action values using the Bellman equation and temporal-difference (TD) updates. It uses a table to store estimates Q(s, a) for each state–action pair. 

### Limitations of Tabular Q-Learning

- Tabular methods only work for **small, discrete state spaces**.
- If the state space is large (e.g., pixel inputs with millions of combinations), maintaining and updating a Q-table becomes infeasible. 

### Deep Q-Learning

Deep Q-Learning uses a **parametrized Q-function** Q(s, a; θ), where a deep neural network (DQN) approximates Q values for all actions given a state. The network generalizes across states, enabling learning in high-dimensional environments. 

Instead of updating Q values in a table, the agent updates network parameters θ using gradient descent on a loss derived from the Bellman equation.

---

## 3.3 — The Deep Q-Network (DQN)

A **Deep Q-Network (DQN)** is a neural network architecture that approximates the Q-function:

- **Input:** Stacked frames representing the state (e.g., last 4 grayscale frames in Atari).  
- **Output:** Q-value for each possible action. 

### Preprocessing and Temporal Information

Because single frames lack motion information, several frames are stacked together to capture temporal dynamics (e.g., ball movement in Pong).  
The frames are typically:

- resized (e.g., to 84×84),
- grayscale (reducing from RGB),  
- stacked in a sequence (e.g., 4 frames deep). 

### Network Architecture

The stacked frames are fed into convolutional layers that learn spatial and temporal features. The network ends with fully connected layers outputting a vector of Q-values. The components of the DQN architecture is shown below:

* **Input:** The state (e.g., 4 frames of a game to capture motion).
* **Hidden Layers:** Convolutional layers (if the input is visual) or Fully Connected layers.
* **Output:** A vector of Q-values, one for each possible action in that state.

---

## 3.4 — The Deep Q-Learning Algorithm

Deep Q-Learning extends Q-Learning using a neural network, but this introduces instability due to:

- Nonlinear function approximation.
- Bootstrapping updates (targets depend on the same network). 

To address this, the DQN algorithm incorporates three key techniques:

### 1. Experience Replay

Instead of learning from consecutive experiences (s, a, r, s′), the agent stores experiences in a **replay buffer** and samples **mini-batches randomly** for training. This:

- improves data efficiency,
- breaks correlation between sequential samples,
- reduces catastrophic forgetting. 

### 2. Fixed Q-Target (Target Network)

Using the same network for both prediction and target estimation causes instability because both network predictions and targets shift simultaneously.

Solution:

- maintain a **target network** (copy of the DQN),
- update target network weights less frequently,
- compute targets using target network to stabilize updates. 

### 3. Double Deep Q-Learning (Double DQN)

Standard DQN tends to **overestimate Q-values** because it uses the maximum estimated Q for the next state. Double DQN mitigates this by decoupling action selection from target value evaluation:

- Use the main network to select the argmax action,
- Use the target network to evaluate its Q-value. 

### Summary of Training Loop

1. Interact with the environment, store experiences in replay buffer.  
2. At update time, sample a mini-batch from the buffer.  
3. Compute predicted Q-values and stable targets.  
4. Minimize loss (e.g., mean squared TD error) by gradient descent.

---

## 3.5 — Glossary of Deep Q-Learning Terms

### Reinforcement Learning Basics
- **Action-Value Function (Q-Function):** Estimates quality (expected future return) of taking action a in state s.  
- **Model-Free:** Does not use a model of environment dynamics (transition probabilities or reward function).

### Value Function Terms (DQN-Specific)
- **Deep Q-Learning:** Uses neural networks to approximate Q-values for environments with large state spaces.
- **Deep Q-Network (DQN):** A neural network that predicts Q(s, a) for all possible actions given a state input. 
- **Experience Replay:** Memory storing transitions (s, a, r, s′) used for random mini-batch training to reduce correlation and improve efficiency. 
- **Target Network:** A separate neural network used to compute stable target Q-values, periodically synced with the main network. 
- **Double DQN:** Variation of DQN that uses two networks to reduce Q-value overestimation. 
- **Temporal Limitation:** Single frames lack motion information; hence, multiple frames are stacked for DQN input.

---

