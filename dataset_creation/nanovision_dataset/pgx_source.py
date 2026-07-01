from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from nanovision_dataset.grayscale import project_to_grayscale
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
        key, init_key = jax.random.split(key)
        state = environment.init(init_key)

        frames: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminals: list[bool] = []

        for _ in range(self.max_steps):
            observation = np.asarray(jax.device_get(state.observation))
            frames.append(project_to_grayscale(observation))
            logits, _value = model(state.observation[None, ...])
            action = _select_action(jnp, logits[0], state.legal_action_mask)
            key, step_key = jax.random.split(key)
            state = environment.step(state, jnp.int32(action), step_key)
            actions.append(action)
            rewards.append(float(np.asarray(jax.device_get(state.rewards))[0]))
            done = bool(np.asarray(jax.device_get(state.terminated | state.truncated)))
            terminals.append(done)
            if done:
                break

        return EpisodeRecord(
            game=game,
            episode=episode,
            seed=seed,
            frames=np.asarray(frames, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int16),
            rewards=np.asarray(rewards, dtype=np.float32),
            terminals=np.asarray(terminals, dtype=bool),
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
