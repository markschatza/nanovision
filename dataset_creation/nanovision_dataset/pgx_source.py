from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nanovision_dataset.minatar_source import EpisodeRecord, normalize_games


@dataclass(frozen=True)
class PgxRuntime:
    rollout_batch: object


class PgxBaselineSource:
    def __init__(
        self,
        max_steps: int = 1000,
        baseline_dir: str = "artifacts/pgx-baselines",
        jax_cache_dir: str | None = "artifacts/jax-cache",
        batch_size: int = 16,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.max_steps = max_steps
        self.baseline_dir = baseline_dir
        self.jax_cache_dir = jax_cache_dir
        self.batch_size = batch_size
        self.policy_source = "pgx-baseline"
        self._runtime_by_game: dict[str, PgxRuntime] = {}

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
            game_seed = seed + game_index * episodes
            records.extend(self.rollout_game(game, episodes=episodes, first_episode=0, seed=game_seed))
        return records

    def rollout_game(self, game: str, episodes: int, first_episode: int, seed: int) -> list[EpisodeRecord]:
        if episodes < 1:
            raise ValueError("episodes must be positive")
        records: list[EpisodeRecord] = []
        for batch_start in range(0, episodes, self.batch_size):
            current_batch_size = min(self.batch_size, episodes - batch_start)
            seeds = np.arange(seed + batch_start, seed + batch_start + current_batch_size, dtype=np.int64)
            records.extend(
                self._rollout_episode_batch(
                    game=game,
                    first_episode=first_episode + batch_start,
                    seeds=seeds,
                )
            )
        return records

    def rollout_episode(self, game: str, episode: int, seed: int) -> EpisodeRecord:
        return self._rollout_episode_batch(game=game, first_episode=episode, seeds=np.asarray([seed], dtype=np.int64))[0]

    def _rollout_episode_batch(self, game: str, first_episode: int, seeds: np.ndarray) -> list[EpisodeRecord]:
        jax, jnp, runtime = self._runtime_for_game(game)
        seed_array = jnp.asarray(seeds, dtype=jnp.uint32)
        keys = jax.vmap(jax.random.PRNGKey)(seed_array)
        batch_frames, batch_actions, batch_rewards, batch_terminals = runtime.rollout_batch(keys)
        batch_frames = np.asarray(jax.device_get(batch_frames), dtype=np.float32)
        batch_actions = np.asarray(jax.device_get(batch_actions), dtype=np.int16)
        batch_rewards = np.asarray(jax.device_get(batch_rewards), dtype=np.float32)
        batch_terminals = np.asarray(jax.device_get(batch_terminals), dtype=bool)

        records: list[EpisodeRecord] = []
        for batch_index, seed in enumerate(seeds):
            frames, actions, rewards, terminals = _trim_episode_arrays(
                batch_frames[batch_index],
                batch_actions[batch_index],
                batch_rewards[batch_index],
                batch_terminals[batch_index],
            )
            records.append(
                EpisodeRecord(
                    game=game,
                    episode=first_episode + batch_index,
                    seed=int(seed),
                    frames=frames,
                    actions=actions,
                    rewards=rewards,
                    terminals=terminals,
                )
            )
        return records

    def _runtime_for_game(self, game: str):
        runtime = self._runtime_by_game.get(game)
        jax, jnp = _load_jax_stack(self.jax_cache_dir)
        if runtime is None:
            pgx = _load_pgx()
            environment = pgx.make(f"minatar-{game}")
            model = pgx.make_baseline_model(f"minatar-{game}_v0", download_dir=self.baseline_dir)
            rollout_episode = _build_rollout_episode_jax(jax, jnp, environment, model, self.max_steps)
            rollout_batch = jax.jit(jax.vmap(rollout_episode))
            runtime = PgxRuntime(rollout_batch=rollout_batch)
            self._runtime_by_game[game] = runtime
        return jax, jnp, runtime


def _trim_episode_arrays(
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    terminal_indices = np.flatnonzero(terminals)
    if terminal_indices.size:
        frame_count = int(terminal_indices[0]) + 1
        frames = frames[:frame_count]
        actions = actions[:frame_count]
        rewards = rewards[:frame_count]
        terminals = terminals[:frame_count]
    return frames, actions, rewards, terminals


def _load_pgx_stack(jax_cache_dir: str | None = "artifacts/jax-cache"):
    jax, jnp = _load_jax_stack(jax_cache_dir)
    return _load_pgx(), jax, jnp


def _load_pgx():
    try:
        import pgx
    except ImportError as exc:
        raise RuntimeError("Pgx baseline policy requires `pgx`, `jax`, and `dm-haiku` dependencies") from exc
    return pgx


def _load_jax_stack(jax_cache_dir: str | None = "artifacts/jax-cache"):
    if jax_cache_dir:
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", jax_cache_dir)
    try:
        import jax
        import jax.numpy as jnp
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
    return jax, jnp


def _episode_seed_ranges(games: Iterable[str] | None, episodes: int, seed: int) -> dict[str, np.ndarray]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    ranges: dict[str, np.ndarray] = {}
    for game_index, game in enumerate(normalize_games(games)):
        start = seed + game_index * episodes
        ranges[game] = np.arange(start, start + episodes, dtype=np.int64)
    return ranges


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


def _build_rollout_episode_jax(jax, jnp, environment, model, max_steps: int):
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

    return functools.partial(rollout, step_count=max_steps)


def _rollout_episode_jax(jax, jnp, environment, model, key, max_steps: int):
    return _build_rollout_episode_jax(jax, jnp, environment, model, max_steps)(key)
