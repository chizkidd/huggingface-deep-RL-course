# Unit 3b: Optuna Hyperparameter Tuning

## Table of Contents
1. [Introduction to Hyperparameter Tuning](#1-introduction-to-hyperparameter-tuning)
2. [Why Hyperparameter Tuning Matters in Deep RL](#2-why-hyperparameter-tuning-matters-in-deep-rl)
3. [Optuna Framework Overview](#3-optuna-framework-overview)
4. [Core Components of Hyperparameter Optimization](#4-core-components-of-hyperparameter-optimization)
5. [Search Strategies](#5-search-strategies)
6. [Pruning Strategies](#6-pruning-strategies)
7. [Implementing Optuna with Stable-Baselines3](#7-implementing-optuna-with-stable-baselines3)
8. [Practical Guidelines and Best Practices](#8-practical-guidelines-and-best-practices)
9. [Advanced Topics](#9-advanced-topics)
10. [Resources and References](#10-resources-and-references)

---

## 1. Introduction to Hyperparameter Tuning

### What is Hyperparameter Tuning?

Hyperparameter tuning is the process of finding the optimal configuration of hyperparameters that maximizes the performance of a machine learning or reinforcement learning model. In Deep RL, this is one of the most critical tasks that can dramatically impact agent performance.

### The Challenge

Finding good hyperparameters manually is:
- **Time-consuming**: Testing different combinations requires multiple training runs
- **Expertise-dependent**: Requires deep understanding of algorithm behavior
- **Resource-intensive**: Each trial consumes computational resources
- **Non-intuitive**: The relationship between hyperparameters and performance is often non-linear and complex

### Example Problem Space

Consider training a neural network with just these 8 hyperparameters:
- Number of hidden layers
- Number of nodes per layer
- Learning rate
- Batch size
- Activation function
- Optimizer type
- Dropout rate
- Weight initialization

Even with just 2 candidate values per hyperparameter: **2^8 = 256 experiments**!

---

## 2. Why Hyperparameter Tuning Matters in Deep RL

### Impact on Performance

Default hyperparameters often don't work well across different environments. For example:
- **PPO on Pendulum-v1**: With default hyperparameters and 100,000 timesteps, PPO achieves ~-1000 average return
- **After tuning**: Can achieve ~-200 average return (near-optimal performance)
- This represents a **5x improvement** in performance!

### Key RL Hyperparameters

Different algorithms have different critical hyperparameters:

#### For DQN (Deep Q-Network):
- Learning rate
- Discount factor (γ)
- Experience replay buffer size
- Batch size
- Epsilon decay schedule
- Target network update frequency (τ)
- Network architecture (hidden layers, nodes)

#### For PPO (Proximal Policy Optimization):
- Learning rate
- Number of steps per rollout (n_steps)
- Discount factor (γ)
- GAE lambda (λ)
- Clip range (ε)
- Number of epochs
- Batch size
- Activation function

#### For SAC (Soft Actor-Critic):
- Learning rate
- Discount factor (γ)
- Replay buffer size
- Train frequency
- Gradient steps
- Tau (soft update coefficient)
- Target entropy

### The Resource Tradeoff

When tuning hyperparameters, you face a fundamental tradeoff:
- **Budget (B)**: Total computational resources available
- **Number of configurations (n)**: How many hyperparameter sets to try
- **Budget per configuration (B/n)**: Resources allocated to each trial

**Goal**: Allocate resources wisely to find optimal hyperparameters within budget constraints.

---

## 3. Optuna Framework Overview

### What is Optuna?

Optuna is an automatic hyperparameter optimization framework designed specifically for machine learning. It was introduced in the paper "Optuna: A Next-generation Hyperparameter Optimization Framework" (KDD 2019).

### Key Features

1. **Define-by-Run API**: Dynamic construction of search spaces using familiar Python syntax
2. **Efficient Algorithms**: State-of-the-art samplers (TPE, CMA-ES) and pruners
3. **Easy Integration**: Works with any ML/DL framework (PyTorch, TensorFlow, Stable-Baselines3)
4. **Parallelization**: Scale to multiple workers with minimal code changes
5. **Visualization**: Built-in plotting and dashboard for optimization history
6. **Database Support**: Persistent storage using SQLite or other databases
7. **Minimal Setup**: Few dependencies and simple installation

### Installation

```bash
pip install optuna
pip install optuna-dashboard  # For visualization dashboard
```

### Basic Workflow

The Optuna workflow consists of three main steps:

```python
import optuna

# 1. Define an objective function to minimize/maximize
def objective(trial):
    # Suggest hyperparameter values
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

# 2. Create a study object
study = optuna.create_study(direction='minimize')

# 3. Optimize over multiple trials
study.optimize(objective, n_trials=100)

# Get best parameters
print(study.best_params)  # E.g., {'x': 2.002108042}
print(study.best_value)   # E.g., 4.45e-06
```

---

## 4. Core Components of Hyperparameter Optimization

### The Two Main Components

Hyperparameter optimization has two key components that manage the exploration-exploitation tradeoff:

1. **Sampler**: Decides which configuration to try next
2. **Pruner**: Decides when to stop unpromising trials early

### 4.1 The Sampler (Search Algorithm)

The sampler determines which hyperparameter values to test next. It explores the search space intelligently.

#### Types of Samplers:

**Random Sampler**
```python
from optuna.samplers import RandomSampler
sampler = RandomSampler()
```
- Samples uniformly from the search space
- Simple baseline but inefficient
- Better than grid search for high-dimensional spaces

**TPE Sampler (Tree-structured Parzen Estimator)**
```python
from optuna.samplers import TPESampler
sampler = TPESampler(n_startup_trials=5)
```
- **Default in Optuna**
- Uses Bayesian optimization approach
- Builds a probabilistic model of the objective function
- More efficient than random search
- Good balance between exploration and exploitation

**CMA-ES Sampler (Covariance Matrix Adaptation Evolution Strategy)**
```python
from optuna.samplers import CmaEsSampler
sampler = CmaEsSampler()
```
- Evolution-based strategy
- Optimizes a population of solutions
- Good for continuous optimization problems
- Can be more effective than TPE for certain problems

### 4.2 The Pruner (Scheduler)

The pruner identifies and stops poorly performing trials early to save computational resources.

#### Types of Pruners:

**Median Pruner**
```python
from optuna.pruners import MedianPruner
pruner = MedianPruner(
    n_startup_trials=5,      # Don't prune first 5 trials
    n_warmup_steps=10        # Wait 10 evaluation steps before pruning
)
```
- Prunes if trial's performance is worse than the median of all trials
- Simple and effective
- Most commonly used

**Hyperband Pruner**
```python
from optuna.pruners import HyperbandPruner
pruner = HyperbandPruner()
```
- Based on successive halving algorithm
- Dynamically allocates resources
- More sophisticated resource allocation

**Pruning Considerations:**
- **Too aggressive**: May discard promising trials that start slowly
- **Too conservative**: Wastes resources on bad configurations
- **Balance**: Use warmup steps to give trials a fair chance

---

## 5. Search Strategies

### Grid Search (Not Recommended)

**Approach**: Discretize search space and try all combinations

**Problems:**
- Exponential growth with dimensions
- May miss optimal regions if discretization is coarse
- Wastes resources on unimportant parameters
- Does not scale

**Example:**
```python
# Grid search - DON'T DO THIS
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]
# Total: 3 × 3 = 9 trials
```

### Random Search (Better Baseline)

**Approach**: Sample hyperparameters uniformly from search space

**Advantages:**
- Better than grid search for high-dimensional spaces
- Can find good solutions by chance
- Easy to parallelize

**Disadvantages:**
- No learning from previous trials
- Inefficient use of resources

```python
import random
learning_rate = random.uniform(1e-5, 1e-2)
batch_size = random.choice([32, 64, 128, 256])
```

### Bayesian Optimization (Recommended)

**Approach**: Build a probabilistic model (surrogate) of the objective function

**How it Works:**
1. Start with random sampling (exploration)
2. Build a surrogate model of objective function
3. Use model to predict promising regions
4. Sample from promising regions (exploitation)
5. Update surrogate model with new results
6. Repeat

**Key Algorithms:**

#### Tree-structured Parzen Estimator (TPE)
- Models P(x|y) rather than P(y|x)
- Separates observations into "good" and "bad" groups
- Samples from distribution more likely to yield good results
- Computationally efficient

#### Gaussian Process (GP)
- Models the objective function directly
- Provides uncertainty estimates
- Can be computationally expensive for large datasets

**Visual Intuition:**
- **Red region**: High performance area (goal)
- **Blue region**: Low performance area
- **Search evolution**: Start broad → Focus on promising regions → Converge to optimum

---

## 6. Pruning Strategies

### Why Pruning?

Pruning identifies poorly performing trials early and stops them to:
- Save computational resources
- Allow more trials within budget
- Focus resources on promising configurations

### When to Prune?

This is the critical decision:
- **Too early**: Can't judge trial quality yet, may discard good configurations
- **Too late**: Wastes resources on bad trials
- **Just right**: Balance between evaluation time and resource efficiency

### Pruning Decision Making

```python
def should_prune(trial_performance, all_trials_performance, warmup_steps):
    """
    Pseudo-code for pruning logic
    """
    if current_step < warmup_steps:
        return False  # Don't prune during warmup
    
    if trial_performance < median(all_trials_performance):
        return True  # Prune if below median
    
    return False
```

### Median Pruner Example

```python
from optuna.pruners import MedianPruner

pruner = MedianPruner(
    n_startup_trials=5,           # Don't prune first 5 trials (need baseline)
    n_warmup_steps=N_EVALUATIONS // 3,  # Wait for 1/3 of evaluations
    interval_steps=1              # Check every evaluation
)
```

**Parameters Explained:**
- `n_startup_trials`: Number of initial trials to complete without pruning (establishes baseline)
- `n_warmup_steps`: Number of steps before pruning can begin (gives trial time to show potential)
- `interval_steps`: How often to check for pruning

### Example: Progressive Pruning

Consider 100 trials with budget of 10,000 steps each:

**Without Pruning:**
- 100 trials × 10,000 steps = 1,000,000 total steps
- All trials run to completion

**With Pruning (Median Pruner):**
- First 5 trials: Complete (startup trials)
- Trial 6-100: Can be pruned at 3,333 steps (after warmup)
- If 50% are pruned at 3,333 steps:
  - 50 trials × 10,000 steps = 500,000 steps
  - 50 trials × 3,333 steps = 166,650 steps
  - **Total: 666,650 steps (33% savings)**
- Use saved budget for additional trials!

---

## 7. Implementing Optuna with Stable-Baselines3

### Complete Implementation Example

Here's a full implementation of Optuna with Stable-Baselines3 for PPO on Pendulum:

#### Step 1: Define Search Space

```python
import optuna
from torch import nn

def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    Sample hyperparameters for PPO algorithm.
    """
    # Sample n_steps as power of 2 (GPU optimization)
    # From 2**5=32 to 2**12=4096
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    n_steps = 2 ** n_steps_pow
    
    # Sample discount factor (closer to 1 is usually better)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    
    # Sample learning rate (log scale is important!)
    learning_rate = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    
    # Sample activation function
    activation_fn_name = trial.suggest_categorical(
        "activation_fn", ["tanh", "relu"]
    )
    
    # Store actual values as user attributes (for reporting)
    trial.set_user_attr("n_steps", n_steps)
    
    # Convert activation function name to PyTorch class
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
    
    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }
```

**Key Points:**
- Use **log scale** for learning rate (values span orders of magnitude)
- Use **powers of 2** for batch sizes and n_steps (GPU optimization)
- Store readable values with `set_user_attr()` for better reporting

#### Step 2: Create Evaluation Callback

```python
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import optuna

class TrialEvalCallback(BaseCallback):
    """
    Callback for evaluating agent during training and reporting to Optuna.
    Also handles pruning of unpromising trials.
    """
    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        
        self.eval_idx = 0
        self.is_pruned = False
        self.last_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """Called at every step"""
        # Check if it's time to evaluate
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            # Calculate mean reward
            mean_reward = np.mean(episode_rewards)
            self.last_mean_reward = mean_reward
            
            # Report to Optuna
            self.trial.report(mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Stop training
        
        return True  # Continue training
```

#### Step 3: Define Objective Function

```python
import gym
from stable_baselines3 import PPO

# Configuration
ENV_NAME = "Pendulum-v1"
N_TIMESTEPS = 100_000
N_EVAL_EPISODES = 5
EVAL_FREQ = 10_000
N_EVALUATIONS = N_TIMESTEPS // EVAL_FREQ

# Default hyperparameters (will be updated)
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "env": ENV_NAME,
    "verbose": 0,
}

def objective(trial: optuna.Trial) -> float:
    """
    Objective function to be maximized by Optuna.
    
    Args:
        trial: Optuna trial object
    
    Returns:
        Mean reward over evaluation episodes
    """
    # Sample hyperparameters
    sampled_params = sample_ppo_params(trial)
    
    # Update default hyperparameters
    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sampled_params)
    
    # Create environments
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    
    # Create the model
    model = PPO(**kwargs)
    
    # Create evaluation callback
    eval_callback = TrialEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        deterministic=True,
    )
    
    # Train the model
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf')
    
    # Handle pruned trials
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    # Return final mean reward
    return eval_callback.last_mean_reward
```

#### Step 4: Run Optimization Study

```python
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Create sampler and pruner
sampler = TPESampler(n_startup_trials=5)
pruner = MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=N_EVALUATIONS // 3
)

# Create study with database storage (recommended)
study = optuna.create_study(
    study_name="ppo_pendulum",
    storage="sqlite:///optuna_ppo.db",  # Persist results
    sampler=sampler,
    pruner=pruner,
    direction="maximize",  # Maximize reward
    load_if_exists=True,   # Resume if study exists
)

# Run optimization
N_TRIALS = 50
TIMEOUT = 3600  # 1 hour timeout (optional)

study.optimize(
    objective,
    n_trials=N_TRIALS,
    timeout=TIMEOUT,
    n_jobs=1,  # Parallel trials (use more if resources allow)
)

# Get best trial
print("Number of finished trials:", len(study.trials))
print("\nBest trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print(f"  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Save results to CSV
study.trials_dataframe().to_csv("study_results_ppo.csv")
```

#### Step 5: Visualize Results

```python
import optuna.visualization as vis

# Optimization history
fig = vis.plot_optimization_history(study)
fig.write_html("optimization_history.html")

# Parameter importances
fig = vis.plot_param_importances(study)
fig.write_html("param_importances.html")

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.write_html("parallel_coordinate.html")

# Slice plot (parameter relationships)
fig = vis.plot_slice(study)
fig.write_html("slice_plot.html")

# Contour plot (for 2D relationships)
fig = vis.plot_contour(study, params=["learning_rate", "gamma"])
fig.write_html("contour_plot.html")
```

#### Step 6: Use Optuna Dashboard (Recommended)

```bash
# Launch dashboard
optuna-dashboard sqlite:///optuna_ppo.db
```

Then open browser to `http://localhost:8080`

The dashboard provides:
- Real-time optimization progress
- Interactive parameter importance plots
- Trial history and details
- Hyperparameter relationships
- Best trial visualization

---

## 8. Practical Guidelines and Best Practices

### 8.1 Start Simple

**Rule #1: Don't optimize too many hyperparameters at once**

Start with a minimal search space and expand only as needed:

```python
# ✅ Good: Start with 2-3 key hyperparameters
def sample_params_simple(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
    }

# ❌ Bad: Trying to optimize everything at once
def sample_params_complex(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "n_steps": trial.suggest_int("n_steps", 32, 4096),
        "batch_size": trial.suggest_int("batch_size", 16, 512),
        "n_epochs": trial.suggest_int("n_epochs", 3, 30),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 0.9),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        # ... too many!
    }
```

### 8.2 Defining Search Ranges

**Start small, expand if needed:**

```python
def sample_params(trial):
    # Start with narrow ranges
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    
    # If best trials saturate at boundaries, expand:
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    return {"learning_rate": learning_rate}
```

**How to decide on ranges:**
1. Look at best trials from initial study
2. If best values are near search space limits → **expand range**
3. If values above/below threshold always fail → **reduce range**

### 8.3 Use Log Scale for Learning Rates

Learning rates span orders of magnitude, so use log scale:

```python
# ✅ Good: Log scale
learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
# Samples: 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2

# ❌ Bad: Linear scale
learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01)
# Heavily biased toward larger values
```

### 8.4 Use Powers of 2 for Discrete Parameters

GPU hardware is optimized for powers of 2:

```python
# ✅ Good: Powers of 2
n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
n_steps = 2 ** n_steps_pow  # 32, 64, 128, 256, ..., 4096

batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

# ❌ Less optimal: Arbitrary values
n_steps = trial.suggest_int("n_steps", 30, 4100)  # Any integer
```

### 8.5 Store Readable Values

Make results easier to interpret:

```python
def sample_params(trial):
    n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
    n_steps = 2 ** n_steps_pow
    
    # Store actual value for easy reading
    trial.set_user_attr("n_steps", n_steps)
    
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]
    
    return {
        "n_steps": n_steps,
        "policy_kwargs": {"activation_fn": activation_fn}
    }
```

### 8.6 Use Database Storage

Always save results to database for:
- Persistence across sessions
- Parallel optimization
- Analysis and visualization

```python
# ✅ Good: Database storage
study = optuna.create_study(
    study_name="my_study",
    storage="sqlite:///optuna.db",
    load_if_exists=True,
)

# ❌ Bad: In-memory only (lost on crash)
study = optuna.create_study()
```

### 8.7 Parallel Optimization

Run multiple trials in parallel:

```python
# Option 1: Multiple processes with shared database
study.optimize(objective, n_trials=50, n_jobs=4)

# Option 2: Multiple machines/containers pointing to same database
# Machine 1:
study = optuna.create_study(storage="postgresql://db_url")
study.optimize(objective, n_trials=25)

# Machine 2:
study = optuna.create_study(storage="postgresql://db_url")
study.optimize(objective, n_trials=25)
```

### 8.8 Monitor Progress

Use visualization to guide the search:

```python
# During optimization, check:
# 1. Optimization history - is it improving?
fig = vis.plot_optimization_history(study)

# 2. Parameter importances - which parameters matter?
fig = vis.plot_param_importances(study)

# 3. Are best parameters at search boundaries? → Expand range
```

### 8.9 Typical Results Timeline

Example timeline for PPO on Pendulum-v1:
- **2 minutes**: First good results emerge
- **5 minutes**: Near-optimal hyperparameters found
- **10+ minutes**: Fine-tuning improvements

**Don't overtune!** Stop when performance plateaus.

### 8.10 Common Pitfalls to Avoid

1. **Too large search space**: Start small, expand if needed
2. **No warmup for pruning**: Give trials time to show potential
3. **Wrong scale for learning rate**: Always use log scale
4. **Not saving to database**: Results lost on crash
5. **Optimizing too early**: Fix bugs in basic implementation first
6. **Over-optimization**: Risk overfitting to specific environment instance

---

## 9. Advanced Topics

### 9.1 Multi-Objective Optimization

Optimize multiple objectives simultaneously:

```python
def objective(trial):
    # ... train model ...
    
    mean_reward = evaluate_reward(model)
    inference_time = measure_inference_time(model)
    
    # Return tuple of objectives
    return mean_reward, -inference_time  # Maximize reward, minimize time

# Create multi-objective study
study = optuna.create_study(
    directions=["maximize", "minimize"]
)
```

### 9.2 Custom Samplers

Implement domain-specific sampling strategies:

```python
from optuna.samplers import BaseSampler

class CustomSampler(BaseSampler):
    def sample_independent(self, study, trial, param_name, param_distribution):
        # Custom sampling logic
        pass
```

### 9.3 Conditional Search Spaces

Create hyperparameter dependencies:

```python
def sample_params(trial):
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    
    if optimizer_name == "adam":
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        beta1 = trial.suggest_float("beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    else:  # SGD
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
```

### 9.4 Integration with Experiment Tracking

Combine Optuna with MLflow or Weights & Biases:

```python
import mlflow

def objective(trial):
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(trial.params)
        
        # ... train model ...
        
        mlflow.log_metric("mean_reward", mean_reward)
    
    return mean_reward
```

### 9.5 Replay Ratio Optimization (Off-Policy Algorithms)

For algorithms like SAC or DQN:

```python
def sample_sac_params(trial):
    # Key insight: Adjust replay ratio for parallel environments
    num_envs = 1024
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 100)
    
    # Replay ratio = gradient_steps / (num_envs * train_freq)
    replay_ratio = gradient_steps / (num_envs * train_freq)
    trial.set_user_attr("replay_ratio", replay_ratio)
    
    return {
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
    }
```

### 9.6 Curriculum Learning Integration

Combine hyperparameter tuning with curriculum learning:

```python
def objective(trial):
    # Sample hyperparameters
    params = sample_params(trial)
    
    # Sample curriculum parameters
    difficulty_increase_rate = trial.suggest_float("curriculum_rate", 0.01, 0.1)
    
    # Train with curriculum
    model = PPO(**params)
    for difficulty in curriculum_schedule(difficulty_increase_rate):
        env.set_difficulty(difficulty)
        model.learn(timesteps_per_level)
    
    return evaluate(model)
```

### 9.7 Architecture Search

Optimize neural network architecture:

```python
def sample_architecture(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    
    net_arch = []
    for i in range(n_layers):
        layer_size = trial.suggest_categorical(
            f"layer_{i}_size",
            [64, 128, 256, 512]
        )
        net_arch.append(layer_size)
    
    return {
        "policy_kwargs": {
            "net_arch": net_arch
        }
    }
```

---

## 10. Resources and References

### Official Documentation
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna GitHub Repository](https://github.com/optuna/optuna)
- [Optuna Examples](https://github.com/optuna/optuna-examples)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

### Tutorials and Guides
- [HuggingFace Deep RL Course - Unit Bonus 2](https://huggingface.co/learn/deep-rl-course/unitbonus2/introduction)
- [Antonin Raffin's Blog - Part 1: Visual Guide](https://araffin.github.io/post/hyperparam-tuning/)
- [Antonin Raffin's Blog - Part 2: Practical Implementation](https://araffin.github.io/post/optuna/)
- [Hands-on RL for Robotics with EAGERx and Stable-Baselines3, ICRA 2022](https://araffin.github.io/tools-for-robotic-rl-icra2022/)
- [ICRA 2022 Tutorial Slides](https://araffin.github.io/slides/icra22-hyperparam-opt/)
- [DataCamp: Optuna for Deep RL](https://www.datacamp.com/tutorial/optuna)

### Video Resources
- ICRA 2022 Tutorial: Automatic Hyperparameter Optimization (Antonin Raffin) [Link](https://www.youtube.com/watch?v=AidFTOdGNFQ)
- Optuna Tutorial Video on YouTube [Link](https://www.youtube.com/watch?v=ihP7E76KGOI)

### Research Papers
- Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework". KDD.
- Henderson et al. (2018). "Deep Reinforcement Learning that Matters". AAAI.

### Related Tools
- **RL Baselines3 Zoo**: Pre-tuned hyperparameters for many environments
- **Optuna Dashboard**: Real-time visualization dashboard
- **OptunaHub**: Feature-sharing platform for Optuna users
- **Neptune.ai**: Experiment tracking integration

### Community Resources
- [GitHub Discussions](https://github.com/optuna/optuna/discussions)
- [HuggingFace Discord](https://discord.gg/aYka4Yhff9)
- Stable-Baselines3 Discussions

---

## Quick Reference: Essential Code Snippets

### Basic Optuna Study

```python
import optuna

def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### Hyperparameter Suggestions

```python
# Float (linear)
lr = trial.suggest_float("lr", 1e-5, 1e-2)

# Float (log scale) - for learning rates
lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

# Integer
n_steps = trial.suggest_int("n_steps", 32, 4096)

# Categorical
activation = trial.suggest_categorical("activation", ["tanh", "relu"])
```

### Complete RL Example Template

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

def sample_params(trial):
    return {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
    }

def objective(trial):
    params = sample_params(trial)
    model = Algorithm(**params)
    model.learn(timesteps)
    return evaluate(model)

study = optuna.create_study(
    storage="sqlite:///optuna.db",
    sampler=TPESampler(n_startup_trials=5),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    direction="maximize"
)

study.optimize(objective, n_trials=50)
print(study.best_params)
```

---

## Summary

Hyperparameter tuning is essential for achieving good performance in Deep RL. Key takeaways:

1. **Manual tuning doesn't scale** - Use automated methods
2. **Optuna is powerful yet simple** - Just 5-6 lines to get started
3. **Start simple** - Optimize 2-3 key parameters first
4. **Use proper scales** - Log scale for learning rates, powers of 2 for batch sizes
5. **Leverage pruning** - Save resources by stopping bad trials early
6. **Monitor progress** - Use visualizations to guide the search
7. **Save results** - Always use database storage
8. **Don't overtune** - Stop when performance plateaus

With these tools and techniques, you can systematically find hyperparameters that work well for your specific RL environment and algorithm.
