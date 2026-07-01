---
name: Nano Pixel RL
last_updated: 2026-07-01
---

# Nano Pixel RL Strategy

## Target problem

RL benchmarks usually test goal-seeking through explicit reward optimization, value functions, or environment-specific policies, which makes it hard to isolate whether sequence prediction alone can produce useful visual control. Contributors need a tiny, hackable benchmark that asks whether a small transformer can infer goals and affordances from visual context, then act with no PPO, reward loss, value head, gradients, or test-time finetuning.

## Our approach

Nano Pixel RL is a nanochat-like benchmark for goal-seeking vision models trained only with cross-entropy on curated visual-control trajectories. The model's language is tokenized visual experience: visual system prompts, current frames, delta-frames, previous visual intents, and actual resulting frames; at test time it plays by emitting the next visual intent while the environment acts as a physics and affordance filter.

The core bet is that well-structured successful trajectories can make goal-directed behavior live implicitly in the data distribution, closer to language-model pretraining or instruction tuning than reinforcement learning. A frozen model should be able to see a short visual prompt for a heldout game or mechanic and infer controllable pixels, dangerous objects, useful contacts, terminal states, and action consequences from context alone.

## Who it's for

**Primary:** Visual-control benchmark contributors - They're hiring Nano Pixel RL to test whether small sequence models can learn in-context goal seeking from visual trajectories without RL-specific training machinery.

**Secondary:** ML tinkerers - They're hiring Nano Pixel RL to edit a small model or loss file, train with one command, and see whether cross-entropy over visual experience produces real heldout game behavior.

## Key metrics

- **Heldout win rate** - Share of heldout games, mechanics, or mechanic compositions solved by the frozen model after a short visual prompt; measured by one-command evaluation rollouts.
- **Time-to-win** - Number of environment steps required to reach a successful terminal state on solved heldout tasks; measured by evaluation logs.
- **Prompt robustness** - Variance in heldout performance across different visual prompts for the same game or mechanic; measured by prompt-sweep evals.
- **Visual-intent validity** - Share of emitted visual intents accepted by the physics and affordance filter without major correction; measured by the environment interpreter.
- **Token efficiency** - Prompt and context tokens required to reach a target heldout score; measured by eval configurations and rollout logs.

## Tracks

### Visual-Control Dataset

Build fixed expert or near-expert trajectory corpora across many tiny synthetic pixel games, including multiple winning strategies, starting states, interactions, edge cases, and limited recovery behavior.

_Why it serves the approach:_ The benchmark only tests sequence-prediction goal seeking if the training distribution contains successful prompted behavior without relying on random exploration-heavy RL logs.

### Prompted Visual Language

Define the tokenizer, episode format, visual system prompts, frame and delta-frame tokens, visual intents, and masked assistant-style loss.

_Why it serves the approach:_ A nanochat-like framing makes the task about predicting the next useful visual intent from context rather than optimizing a scalar reward during training.

### Physics And Affordance Filter

Keep the environment responsible for projecting proposed visual intents into valid next states, so legal paddle movement, collisions, pickups, walls, and object constraints are enforced outside the model.

_Why it serves the approach:_ The model can express desired visual changes while the benchmark stays grounded in game mechanics instead of rewarding arbitrary pixel edits.

### Heldout Evaluation Harness

Lock down one-command train and eval flows with fixed tiny games, fixed tokenizer, fixed datasets, fixed heldout splits, and a small reference model.

_Why it serves the approach:_ Credible comparison depends on evaluating frozen models on withheld games, mechanics, or mechanic compositions with no gradients and no training-time scalar rewards.

## Not working on

- PPO, value functions, reward losses, or test-time finetuning; the point is cross-entropy over visual-control trajectories.
- Random exploration-heavy RL log generation as the main data source; curated successful trajectories are the core training signal.
- Arbitrary pixel editing; the environment remains a physics and affordance filter.
- A broad Atari-scale platform for v1; start with tiny hackable 16x16 or 32x32 games such as Pong, Breakout, dodge, collect, maze, key-door, and avoid-enemy.

## Marketing

**One-liner:** A nanochat-like benchmark where a small transformer learns to play tiny visual games from cross-entropy over prompted visual-control trajectories.

**Key message:** The long-term thesis is that cross-entropy over well-structured visual-control data can produce a small but real form of in-context goal seeking: a frozen model sees a visual prompt, infers the goal and affordances, and plays by emitting visual intents.
