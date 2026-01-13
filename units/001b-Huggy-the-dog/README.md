# Bonus Unit 1: Intro to Deep RL with Huggy 

In this bonus unit, we work with **Huggy the Dog**, an environment created in **Unity.** Unlike the simple grids of FrozenLake, this is a 3D environment where we train a dog to fetch a stick using **Deep Reinforcement Learning**.

## 1 Introduction

The goal of this unit is to see Deep RL in a more "lifelike" setting.

* **The Agent:** Huggy the Dog.
* **The Task:** Fetch the stick and bring it back to the target.
* **The Tool:** We use **Unity ML-Agents**, a toolkit that allows us to use Unity environments as RL gyms.



## 2 How Huggy Works

Huggy is a Deep Reinforcement Learning environment made by Hugging Face and based on Puppo the Corgi, a project by the Unity MLAgents team. This environment was created using the Unity game engine and MLAgents. 
ML-Agents is a toolkit for the game engine from Unity that allows us to create environments using Unity or use pre-made environments to train our agents.
In this environment, we aim to train Huggy to **fetch the stick we throw. This means he needs to move correctly toward the stick.**

Huggy isn't just a 3D model; he is a complex RL agent with specific inputs and outputs. 

### The Observation Space (What Huggy "perceives")

Huggy doesn't see "pixels." Instead, he receives a **Vector Observation** (a list of numbers that are information about the environment):

1. **Target Position:** Where the stick/target is located.
2. **Relative Position:** Where Huggy is in relation to the target.
3. **Orientation of His Legs:** The current rotation of his legs and body parts.

With all this information, Huggy can _use his policy to determine which action to take next to fulfill his goal._

### The Action Space (What Huggy "does")

Huggy uses a **Continuous Action Space**.
* Instead of "Left" or "Right," the brain outputs numbers (floats) that represent **Torque (force)** applied to his leg joints.
* Joint motors drive Huggy’s legs. This means that to get the target, Huggy needs to ***learn to rotate the joint motors of each of his legs correctly so he can move.***
* *Learning to walk/run is the first sub-task the agent implicitly solves.*

### The Reward Function (Motivation)
The goal is that Huggy moves towards the stick without spinning too much.

* **Positive Reward:** Getting closer to the stick.
* **Positive Reward:** Touching the stick.
* **Negative Reward:** Spinning too much and turning too quickly.
* **Negative Reward:** Time penalty-> a fixed-time penalty given at every action to force him to get to the stick as fast as possible.


## 3 Training Huggy

Huggy aims to ***learn to run correctly and as fast as possible toward the goal.*** To do that, at every step and given the environment observation, 
he needs to decide how to rotate each joint motor of his legs to move correctly (not spinning too much) and towards the goal. 
Because Huggy has a complex continuous action space, we use **PPO (Proximal Policy Optimization)**.

### The Training Process:

1. **The Brain:** We use a Neural Network that takes the vector observations and outputs joint forces.
2. **The Environment:** Unlike `gym`, Unity ML-Agents can run multiple "clones" of the environment simultaneously to speed up data collection.
3. **Hyperparameters:**
* **Batch Size:** How many experiences are used for one update.
* **Learning Rate:** How fast the "brain" changes its mind.
* **Time Horizon:** How many steps the agent looks forward to calculate rewards.

### The Training Loop

```mermaid
graph TD
    %% Nodes
    Agent["<b>Agent (Huggy)</b>"]
    Env["<b>Environment</b>"]

    %% Horizontal flow
    Env -- "<b>Sₜ:</b> Stick & Leg position<br/><b>Rₜ:</b> Reward" --> Agent
    Agent -- "<b>Action Aₜ</b><br/>(Joint movements)" --> Env

    %% Styling for clarity
    style Agent fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Env fill:#f9f9f9,stroke:#333,stroke-width:2px
```

```mermaid
graph TD
    Agent["<b>Agent (Huggy)</b>"]
    Env["<b>Environment</b>"]

    %% Right side flow (Action)
    Agent ---->|<b>Action Aₜ</b><br/>Joint movements| Env

    %% Left side flow (State/Reward)
    Env ---->|<b>Sₜ:</b> Stick & Leg position<br/><b>Rₜ:</b> Reward| Agent

    %% Styling
    style Agent fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Env fill:#f9f9f9,stroke:#333,stroke-width:2px
```

```mermaid
graph TD
    %% Nodes
    Agent["<b>Agent (Huggy)</b>"]
    Env["<b>Environment</b>"]

    %% Right side: Agent to Environment (Downward)
    Agent -- "<b>Action Aₜ</b>" --> Env

    %% Left side: Environment to Agent (Upward)
    Env -- "<b>Sₜ, Rₜ</b>" --> Agent

    %% Styling
    style Agent fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Env fill:#f9f9f9,stroke:#333,stroke-width:2px
```

![RL loop]('../img/huggy_train_loop.svg')


<img src="[https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit1/RL_process_game.jpg](https://cas-bridge.xethub.hf.co/xet-bridge-us/637f72fc62a4445929f4fcb3/168f10f0fb30b8515e33cbcbb8bb5417534257e7f0731b3d72eabd584f282b60?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260113%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260113T203012Z&X-Amz-Expires=3600&X-Amz-Signature=facae8a8220531fb3f3c09859ecd4d6ff5b6a1082c78c1ac24e7183349786f4e&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=638faa7f21d355ca70d54e38&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27huggy-loop.jpg%3B+filename%3D%22huggy-loop.jpg%22%3B&response-content-type=image%2Fjpeg&x-id=GetObject&Expires=1768339812&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc2ODMzOTgxMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82MzdmNzJmYzYyYTQ0NDU5MjlmNGZjYjMvMTY4ZjEwZjBmYjMwYjg1MTVlMzNjYmNiYjhiYjU0MTc1MzQyNTdlN2YwNzMxYjNkNzJlYWJkNTg0ZjI4MmI2MCoifV19&Signature=sgThU1ig6mFzsGqkksAdb6zBP3nAu4sHolcCII1asyScb3emJX-jpz6H%7EiehH-p3eHET5-tdhtNw-9GbwC1WnogSj29xVD53XjHPhKFn%7EFVuKfQS9nK4Ug6-wxqH7kGWChDW0CWrswaMnBBqAYL9vC4lrU7ADkiDpCWVA-r756rwOuCrvj54Cx5irSBmSB1prfegWlz1SCimUHH%7ENZqhJ9Bn9nBGB%7EgOnYEWaGTmB6NI%7EWRH%7E%7EOmCRYWkSszTUtijLYdPJmk6hnQWMKu1RSVvzE1QbevpJ1SfzQ-bDnjotEZ6nc-CvqvUkg%7E%7E98%7ERUhiOidLoWbkjSwOiWpsJwj9cQ__&Key-Pair-Id=K2L8F4GPSG1IFC]" alt="The RL process" width="100%">

## 4 Play and Share

Once training is finished, the model is exported as a `.nn` or `.onnx` file.

* **The Play Loop:** The model is loaded into the Unity executable, and Huggy now "knows" how to run toward the stick based on the weights learned during training.
* **The Hub:** You can upload your Huggy model to the Hugging Face Hub just like you did with LunarLander.

---

### Recommended Repository Update

I suggest adding a "Practices" section to your `units/` folder to keep things clean.

```text
huggingface-deep-rl-study/
├── units/
│   ├── unit1-intro-rl/
│   ├── unit2-q-learning/
│   └── unit-bonus1-huggy/  <-- Add this!
│       ├── 01-intro-to-DeepRL-with-huggy.md
│       └── huggy_ppo_config.yaml  <- For your hyperparameters

```

### Quick Tip for Huggy

This unit uses a **Unity WebGL** demo to let you play with your trained agent in the browser. When you finish training, make sure to grab your **Model ID** (e.g., `Chiz/ppo-Huggy`) so you can load your personal dog in the [Huggy Playable Demo](https://huggingface.co/spaces/ThomasSimonini/Huggy).

