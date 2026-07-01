from __future__ import annotations

import functools
import os
from typing import Iterable

import numpy as np

from nanovision_dataset.minatar_source import EpisodeRecord, normalize_games


class PgxBaselineSource:
    def __init__(
        self,
        max_steps: int = 1000,
        baseline_dir: str = "artifacts/pgx-baselines",
        jax_cache_dir: str | None = "artifacts/jax-cache",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self.baseline_dir = baseline_dir
        self.jax_cache_dir = jax_cache_dir
        self.policy_source = "pgx-baseline"

    def rollout_games(
        self,
        games: Iterable[str] | None,
        episodes: int,
        seed: int,
    ) -> list[EpisodeRecord]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        records: list[EpisodeRecord] = []
        for game_index, game in enumerate(normalize_games(games)):
            for episode in range(episodes):
                episode_seed = seed + game_index * episodes + episode
                records.append(self.rollout_episode(game, episode=episode, seed=episode_seed))
        return records

    def rollout_episode(self, game: str, episode: int, seed: int) -> EpisodeRecord:
        pgx, jax, jnp = _load_pgx_stack(self.jax_cache_dir)
        environment = pgx.make(f"minatar-{game}")
        model = pgx.make_baseline_model(f"minatar-{game}_v0", download_dir=self.baseline_dir)

        key = jax.random.PRNGKey(seed)
        frames, actions, rewards, terminals = _rollout_episode_jax(
            jax,
            jnp,
            environment,
            model,
            key,
            self.max_steps,
        )
        frames = np.asarray(jax.device_get(frames), dtype=np.float32)
        actions = np.asarray(jax.device_get(actions), dtype=np.int16)
        rewards = np.asarray(jax.device_get(rewards), dtype=np.float32)
        terminals = np.asarray(jax.device_get(terminals), dtype=bool)

        terminal_indices = np.flatnonzero(terminals)
        if terminal_indices.size:
            frame_count = int(terminal_indices[0]) + 1
            frames = frames[:frame_count]
            actions = actions[:frame_count]
            rewards = rewards[:frame_count]
            terminals = terminals[:frame_count]

        return EpisodeRecord(
            game=game,
            episode=episode,
            seed=seed,
            frames=frames,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
        )


def _load_pgx_stack(jax_cache_dir: str | None = "artifacts/jax-cache"):
    if jax_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", jax_cache_dir)
    try:
        import jax
        import jax.numpy as jnp
        import pgx
    except ImportError as exc:
        raise RuntimeError("Pgx baseline policy requires `pgx`, `jax`, and `dm-haiku` dependencies") from exc
    if jax_cache_dir:
        jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        _update_jax_config_if_available(
            jax,
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )
    return pgx, jax, jnp


def _update_jax_config_if_available(jax, name: str, value: object) -> None:
    try:
        jax.config.update(name, value)
    except AttributeError:
        pass


def _select_action(jnp, logits, legal_action_mask) -> int:
    masked_logits = jnp.where(legal_action_mask, logits, -jnp.inf)
    return int(jnp.argmax(masked_logits))


def _select_action_array(jnp, logits, legal_action_mask):
    masked_logits = jnp.where(legal_action_mask, logits, -jnp.inf)
    return jnp.argmax(masked_logits).astype(jnp.int32)


def _project_to_grayscale_jax(jnp, observation):
    channel_count = observation.shape[-1]
    if channel_count == 1:
        weights = jnp.asarray([1.0], dtype=jnp.float32)
    else:
        weights = jnp.linspace(0.25, 1.0, channel_count, dtype=jnp.float32)
    active = observation > 0
    weighted = active * weights.reshape((1, 1, -1))
    return weighted.max(axis=-1).astype(jnp.float32)


def _rollout_episode_jax(jax, jnp, environment, model, key, max_steps: int):
    @functools.partial(jax.jit, static_argnums=1)
    def rollout(start_key, step_count: int):
        start_key, init_key = jax.random.split(start_key)
        initial_state = environment.init(init_key)
        initial_done = jnp.asarray(False)

        def scan_step(carry, _):
            state, key, already_done = carry
            frame = _project_to_grayscale_jax(jnp, state.observation)
            logits, _value = model(state.observation[None, ...])
            action = _select_action_array(jnp, logits[0], state.legal_action_mask)
            key, step_key = jax.random.split(key)

            def step_active(_):
                next_state = environment.step(state, action, step_key)
                done = next_state.terminated | next_state.truncated
                reward = next_state.rewards[0]
                return next_state, reward, done

            def step_inactive(_):
                return state, jnp.asarray(0.0, dtype=jnp.float32), already_done

            next_state, reward, done = jax.lax.cond(already_done, step_inactive, step_active, operand=None)
            terminal = already_done | done
            return (next_state, key, terminal), (frame, action, reward.astype(jnp.float32), terminal)

        _carry, outputs = jax.lax.scan(
            scan_step,
            (initial_state, start_key, initial_done),
            xs=None,
            length=step_count,
        )
        return outputs

    return rollout(key, max_steps)
