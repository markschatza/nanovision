import numpy as np
import pytest

from nanovision_dataset.minatar_source import DEFAULT_GAMES, EpisodeRecord, MinAtarSource, normalize_games


def test_default_game_list_contains_five_required_games() -> None:
    assert DEFAULT_GAMES == ("asterix", "breakout", "freeway", "seaquest", "space_invaders")


def test_normalize_games_rejects_unknown_games() -> None:
    with pytest.raises(ValueError):
        normalize_games(["breakout", "pong"])


def test_rollout_games_rejects_non_positive_episodes() -> None:
    source = MinAtarSource(max_steps=1)

    with pytest.raises(ValueError):
        source.rollout_games(["breakout"], episodes=0, seed=0)


def test_episode_record_uses_aligned_arrays() -> None:
    record = EpisodeRecord(
        game="breakout",
        episode=0,
        seed=7,
        frames=np.zeros((2, 10, 10), dtype=np.float32),
        actions=np.asarray([1, 2], dtype=np.int16),
        rewards=np.asarray([0.0, 1.0], dtype=np.float32),
        terminals=np.asarray([False, True]),
    )

    assert len(record.frames) == len(record.actions) == len(record.rewards) == len(record.terminals)


def test_seeded_real_rollouts_are_deterministic() -> None:
    source = MinAtarSource(max_steps=3)

    first = source.rollout_episode("breakout", episode=0, seed=123)
    second = source.rollout_episode("breakout", episode=0, seed=123)

    np.testing.assert_array_equal(first.frames, second.frames)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.rewards, second.rewards)
    np.testing.assert_array_equal(first.terminals, second.terminals)
