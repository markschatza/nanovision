import numpy as np

from nanovision_dataset.grayscale import project_to_grayscale
from nanovision_dataset.pgx_source import (
    PgxBaselineSource,
    _project_to_grayscale_jax,
    _select_action,
    _update_jax_config_if_available,
)


def test_pgx_source_records_jax_cache_dir() -> None:
    source = PgxBaselineSource(max_steps=1, baseline_dir="baselines", jax_cache_dir="cache")

    assert source.jax_cache_dir == "cache"


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
