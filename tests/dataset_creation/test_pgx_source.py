from nanovision_dataset.pgx_source import PgxBaselineSource, _select_action, _update_jax_config_if_available


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
