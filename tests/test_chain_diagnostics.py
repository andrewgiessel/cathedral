"""Reference-oracle checks for rank-normalized split-chain MCMC diagnostics."""

import numpy as np
import pytest

from cathedral.chains import diagnose_chains


def _ar1(seed: int, coefficient: float, chains: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(chains, draws))
    values = np.empty_like(noise)
    values[:, 0] = noise[:, 0]
    for index in range(1, draws):
        values[:, index] = coefficient * values[:, index - 1] + noise[:, index]
    return values


@pytest.fixture(
    params=[
        (
            "iid",
            lambda: np.random.default_rng(12).normal(size=(4, 400)),
            (0.9980320924760665, 1521.408213041405, 1570.6005214211807, 1524.4046342289103, 0.025634542528192875),
        ),
        (
            "ar1",
            lambda: _ar1(31, 0.96, 4, 600),
            (1.0851222229961897, 35.10111118768168, 146.11965730178514, 35.60084826756864, 0.6086211024791787),
        ),
        (
            "negative_ar",
            lambda: _ar1(44, -0.9, 4, 401),
            (1.0049777042664594, 5126.591972249479, 503.5145845444553, 5126.591972249479, 0.0326775804540724),
        ),
        (
            "odd_iid",
            lambda: np.random.default_rng(82).normal(size=(3, 401)),
            (1.000424178963133, 1291.7472001766444, 1167.811227944091, 1287.0247112889258, 0.027561984213154034),
        ),
        (
            "tied_discrete",
            lambda: np.random.default_rng(51).integers(0, 4, size=(4, 400)).astype(float),
            (1.0006922971713466, 1627.7584599291065, 1568.2341917756187, 1630.778913030089, 0.027791242108527986),
        ),
    ]
)
def reference_case(request: pytest.FixtureRequest):
    """Fixed ArviZ 0.23.4 oracle values; generated with NumPy/SciPy current in CI."""
    return request.param


def test_diagnostics_match_verified_arviz_oracles(reference_case):
    _name, build, expected = reference_case
    result = diagnose_chains(build())
    actual = (result.rhat, result.ess_bulk, result.ess_tail, result.ess_mean, result.mcse_mean)
    assert result.status == "ok"
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_shifted_chains_match_arviz_and_flag_nonconvergence():
    draws = np.random.default_rng(6).normal(size=(4, 300))
    draws[:2] += 2.5
    result = diagnose_chains(draws)
    expected = (1.5672747983458368, 7.0650896518807835, 200.97709452127376, 6.525902930069599, 0.6236596510833355)
    assert result.status == "ok"
    np.testing.assert_allclose(
        (result.rhat, result.ess_bulk, result.ess_tail, result.ess_mean, result.mcse_mean),
        expected,
        rtol=2e-12,
        atol=2e-12,
    )
    assert result.rhat > 1.1


def test_optional_live_arviz_agreement(reference_case):
    az = pytest.importorskip("arviz")
    _name, build, _expected = reference_case
    draws = build()
    result = diagnose_chains(draws)
    expected = (
        az.rhat(draws, method="rank"),
        az.ess(draws, method="bulk"),
        az.ess(draws, method="tail"),
        az.ess(draws, method="mean"),
        az.mcse(draws, method="mean"),
    )
    np.testing.assert_allclose(
        (result.rhat, result.ess_bulk, result.ess_tail, result.ess_mean, result.mcse_mean),
        expected,
        rtol=2e-12,
        atol=2e-12,
    )


def test_constants_nonfinite_and_short_inputs_have_explicit_statuses():
    constant = diagnose_chains(np.ones((4, 100)))
    nonfinite = diagnose_chains(np.array([[1.0, 2.0, np.nan, 4.0]] * 2))
    short = diagnose_chains(np.ones((2, 3)))

    assert constant.status == "constant"
    assert np.isnan(constant.mcse_mean)
    assert np.isnan(constant.rhat)
    assert constant.warnings
    assert nonfinite.status == "invalid"
    assert short.status == "invalid"
    assert nonfinite.warnings and short.warnings


def test_invalid_folded_rhat_is_not_masked_by_finite_bulk_rhat():
    # Every value is equally far from the median, so folded rank R-hat is
    # undefined while the un-folded rank transform remains nonconstant.
    result = diagnose_chains(np.tile([0.0, 1.0], (4, 100)))
    assert result.status == "invalid"
    assert np.isnan(result.rhat)
    assert any("undefined" in warning for warning in result.warnings)
