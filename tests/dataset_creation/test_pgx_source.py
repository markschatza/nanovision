import numpy as np

from nanovision_dataset.grayscale import project_to_grayscale
from nanovision_dataset.minatar_source import EpisodeRecord
from nanovision_dataset.pgx_source import (
    PgxBaselineSource,
    _episode_seed_ranges,
    _project_to_grayscale_jax,
    _select_action,
    _select_action_array,
    _trim_episode_arrays,
    _update_jax_config_if_available,
)


def test_pgx_source_records_jax_cache_dir() -> None:
    source = PgxBaselineSource(max_steps=1, baseline_dir="baselines", jax_cache_dir="cache", batch_size=4)

    assert source.jax_cache_dir == "cache"
    assert source.batch_size == 4


def test_pgx_source_rejects_invalid_batch_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="batch_size"):
        PgxBaselineSource(batch_size=0)


def test_update_jax_config_if_available_ignores_old_jax_flags() -> None:
    class StubConfig:
        def update(self, name: str, value: object) -> None:
            raise AttributeError(name)

    class StubJax:
        config = StubConfig()

    _update_jax_config_if_available(StubJax, "newer_flag", "value")


def test_jax_grayscale_projection_matches_numpy_projection() -> None:
    import jax.numpy as jnp

    observation = np.zeros((10, 10, 4), dtype=np.float32)
    observation[0, 0, 0] = 1.0
    observation[1, 1, 2] = 1.0
    observation[2, 2, 3] = 1.0

    frame = _project_to_grayscale_jax(jnp, jnp.asarray(observation))

    np.testing.assert_allclose(np.asarray(frame), project_to_grayscale(observation))


def test_episode_seed_ranges_preserve_game_major_order() -> None:
    ranges = _episode_seed_ranges(["breakout", "seaquest"], episodes=3, seed=100)

    assert list(ranges) == ["breakout", "seaquest"]
    np.testing.assert_array_equal(ranges["breakout"], np.asarray([100, 101, 102]))
    np.testing.assert_array_equal(ranges["seaquest"], np.asarray([103, 104, 105]))


def test_trim_episode_arrays_uses_first_terminal() -> None:
    frames = np.zeros((5, 10, 10), dtype=np.float32)
    actions = np.arange(5, dtype=np.int16)
    rewards = np.arange(5, dtype=np.float32)
    terminals = np.asarray([False, True, True, True, True])

    trimmed = _trim_episode_arrays(frames, actions, rewards, terminals)

    assert [len(array) for array in trimmed] == [2, 2, 2, 2]
    assert trimmed[3].tolist() == [False, True]


def test_trim_episode_arrays_keeps_full_nonterminal_rollout() -> None:
    frames = np.zeros((3, 10, 10), dtype=np.float32)
    actions = np.arange(3, dtype=np.int16)
    rewards = np.arange(3, dtype=np.float32)
    terminals = np.zeros(3, dtype=bool)

    trimmed = _trim_episode_arrays(frames, actions, rewards, terminals)

    assert [len(array) for array in trimmed] == [3, 3, 3, 3]


def test_rollout_games_batches_records_and_preserves_seed_order(monkeypatch) -> None:
    source = PgxBaselineSource(max_steps=4, batch_size=2)
    calls: list[tuple[str, list[int]]] = []

    def fake_batch(game: str, first_episode: int, seeds: np.ndarray):
        calls.append((game, [int(seed) for seed in seeds]))
        records = []
        for offset, seed in enumerate(seeds):
            records.append(
                _record(
                    game=game,
                    episode=first_episode + offset,
                    seed=int(seed),
                    length=1,
                )
            )
        return records

    monkeypatch.setattr(source, "_rollout_episode_batch", fake_batch)

    records = source.rollout_games(["breakout", "seaquest"], episodes=3, seed=100)

    assert [(record.game, record.episode, record.seed) for record in records] == [
        ("breakout", 0, 100),
        ("breakout", 1, 101),
        ("breakout", 2, 102),
        ("seaquest", 0, 103),
        ("seaquest", 1, 104),
        ("seaquest", 2, 105),
    ]
    assert calls == [
        ("breakout", [100, 101]),
        ("breakout", [102]),
        ("seaquest", [103, 104]),
        ("seaquest", [105]),
    ]


def test_runtime_is_cached_per_game(monkeypatch) -> None:
    source = PgxBaselineSource(max_steps=4, batch_size=2)
    created: list[str] = []

    class FakeJax:
        def jit(self, fn):
            return fn

        def vmap(self, fn):
            return fn

    class FakePgx:
        def make(self, env_id: str):
            created.append(env_id)
            return object()

        def make_baseline_model(self, model_id: str, download_dir: str):
            created.append(model_id)
            return object()

    monkeypatch.setattr("nanovision_dataset.pgx_source._load_jax_stack", lambda cache_dir: (FakeJax(), object()))
    monkeypatch.setattr("nanovision_dataset.pgx_source._load_pgx", lambda: FakePgx())
    monkeypatch.setattr(
        "nanovision_dataset.pgx_source._build_rollout_episode_jax",
        lambda jax, jnp, environment, model, max_steps: object(),
    )

    first = source._runtime_for_game("breakout")[2]
    second = source._runtime_for_game("breakout")[2]

    assert first is second
    assert created == ["minatar-breakout", "minatar-breakout_v0"]


def test_batched_select_action_masks_each_row() -> None:
    import jax
    import jax.numpy as jnp

    logits = jnp.asarray([[10.0, 2.0, 3.0], [1.0, 8.0, 7.0]])
    masks = jnp.asarray([[False, True, True], [True, False, True]])
    actions = jax.vmap(lambda row, mask: _select_action_array(jnp, row, mask))(logits, masks)

    np.testing.assert_array_equal(np.asarray(actions), np.asarray([2, 2]))


def test_select_action_masks_illegal_actions() -> None:
    import jax.numpy as jnp

    action = _select_action(
        jnp,
        jnp.asarray([10.0, 1.0, 2.0]),
        jnp.asarray([False, True, True]),
    )

    assert action == 2


def test_select_action_uses_highest_legal_logit() -> None:
    import jax.numpy as jnp

    action = _select_action(
        jnp,
        jnp.asarray([0.0, 3.0, 2.0]),
        jnp.asarray([True, True, True]),
    )

    assert action == 1


def _record(game: str, episode: int, seed: int, length: int):
    return EpisodeRecord(
        game=game,
        episode=episode,
        seed=seed,
        frames=np.zeros((length, 10, 10), dtype=np.float32),
        actions=np.zeros(length, dtype=np.int16),
        rewards=np.zeros(length, dtype=np.float32),
        terminals=np.zeros(length, dtype=bool),
    )
