from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from nanovision_dataset.grayscale import project_to_grayscale


DEFAULT_GAMES: tuple[str, ...] = (
    "asterix",
    "breakout",
    "freeway",
    "seaquest",
    "space_invaders",
)


@dataclass(frozen=True)
class EpisodeRecord:
    game: str
    episode: int
    seed: int
    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray


def normalize_games(games: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(games) if games is not None else DEFAULT_GAMES
    unknown = sorted(set(selected) - set(DEFAULT_GAMES))
    if unknown:
        raise ValueError(f"unknown MinAtar game(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("at least one game is required")
    return selected


class MinAtarSource:
    def __init__(self, max_steps: int = 1000, policy_source: str = "random") -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if policy_source != "random":
            raise ValueError(f"unsupported policy_source: {policy_source}")
        self.max_steps = max_steps
        self.policy_source = policy_source

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
        environment = _create_environment(game)
        rng = np.random.default_rng(seed)
        _reset_environment(environment, seed)

        frames: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminals: list[bool] = []

        for _ in range(self.max_steps):
            state = np.asarray(environment.state())
            frames.append(project_to_grayscale(state))

            action = int(rng.integers(0, _num_actions(environment)))
            reward, terminal = _act(environment, action)
            actions.append(action)
            rewards.append(float(reward))
            terminals.append(bool(terminal))
            if terminal:
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


def _create_environment(game: str):
    try:
        from minatar import Environment
    except ImportError as exc:
        raise RuntimeError("MinAtar is not installed; run `uv sync` before generating datasets") from exc
    return Environment(env_name=game)


def _reset_environment(environment, seed: int) -> None:
    _seed_random_state(environment, seed)
    inner_env = getattr(environment, "env", None)
    if inner_env is not None:
        _seed_random_state(inner_env, seed)
    try:
        environment.reset(seed=seed)
    except TypeError:
        environment.reset()


def _seed_random_state(target, seed: int) -> None:
    random_state = getattr(target, "random", None)
    if random_state is not None and hasattr(random_state, "seed"):
        random_state.seed(seed)


def _num_actions(environment) -> int:
    if hasattr(environment, "num_actions"):
        return int(environment.num_actions())
    if hasattr(environment, "minimal_action_set"):
        return len(environment.minimal_action_set())
    raise RuntimeError("MinAtar environment does not expose an action count")


def _act(environment, action: int) -> tuple[float, bool]:
    result = environment.act(action)
    if isinstance(result, tuple) and len(result) >= 2:
        return float(result[0]), bool(result[1])
    terminal = bool(environment.terminal()) if hasattr(environment, "terminal") else False
    reward = float(result) if result is not None else 0.0
    return reward, terminal
